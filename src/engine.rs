//! Cross-match engine: stream Parquet A/B, HEALPix index, haversine join, write matches.
//! Supports pre-partitioned B (shard directory), n_nearest, parallelism, and progress callback.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayAccessor, DictionaryArray, Float32Array, Float64Array, Int64Array, StringArray,
    UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use cdshealpix::haversine_dist;
use cdshealpix::nested::{self, get};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
const RAD_TO_ARCSEC: f64 = RAD_TO_DEG * 3600.0;

/// Result statistics from cross_match_impl (returned to Python as CrossMatchResult).
#[derive(Clone, Debug)]
pub struct CrossMatchStats {
    pub output_path: String,
    pub rows_a_read: u64,
    pub rows_b_read: u64,
    pub matches_count: u64,
    pub chunks_processed: usize,
    pub time_seconds: f64,
}

/// ID value: int64 or string to preserve schema.
#[derive(Clone, Hash, Eq, PartialEq)]
pub enum IdVal {
    I64(i64),
    Str(String),
}

/// Haversine angular distance in arcsec (inputs in degrees).
fn haversine_arcsec(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let lon1 = ra1_deg * DEG_TO_RAD;
    let lat1 = dec1_deg * DEG_TO_RAD;
    let lon2 = ra2_deg * DEG_TO_RAD;
    let lat2 = dec2_deg * DEG_TO_RAD;
    let rad = haversine_dist(lon1, lat1, lon2, lat2);
    rad * RAD_TO_ARCSEC
}

/// Resolve ID column name: use provided or first non-ra/dec column.
fn id_column(batch: &RecordBatch, ra_col: &str, dec_col: &str, id_col: Option<&str>) -> String {
    if let Some(c) = id_col {
        return c.to_string();
    }
    let schema = batch.schema();
    for i in 0..schema.fields().len() {
        let name = schema.field(i).name();
        if name.to_lowercase() != ra_col && name.to_lowercase() != dec_col {
            return name.clone();
        }
    }
    "index".to_string()
}

/// Infer Arrow DataType for ID column (value type when Dictionary).
fn id_type(batch: &RecordBatch, col_name: &str) -> DataType {
    let schema = batch.schema();
    let field = schema
        .field_with_name(col_name)
        .expect("id column missing");
    match field.data_type() {
        DataType::Dictionary(_, value_type) => (**value_type).clone(),
        other => other.clone(),
    }
}

/// Extract ID value from batch row. Supports Dictionary-encoded columns (Parquet often uses these).
fn get_id_value(batch: &RecordBatch, _col_name: &str, col_idx: usize, row: usize) -> IdVal {
    let col = batch.column(col_idx);
    match col.data_type() {
        DataType::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            IdVal::I64(arr.value(row))
        }
        DataType::Int32 => {
            let arr = col.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            IdVal::I64(arr.value(row) as i64)
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            IdVal::Str(arr.value(row).to_string())
        }
        DataType::Dictionary(_, value_type) => {
            if let Some(dict) = col.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
                match value_type.as_ref() {
                    DataType::Int64 => {
                        let typed = dict.downcast_dict::<Int64Array>().unwrap();
                        IdVal::I64(typed.value(row))
                    }
                    DataType::Utf8 | DataType::LargeUtf8 => {
                        let typed = dict.downcast_dict::<StringArray>().unwrap();
                        IdVal::Str(typed.value(row).to_string())
                    }
                    _ => IdVal::Str(format!("{:?}", value_type)),
                }
            } else {
                IdVal::Str(format!("{:?}", col.as_any()))
            }
        }
        _ => IdVal::Str(format!("{:?}", col.as_any())),
    }
}

/// Column index by name.
fn column_index(batch: &RecordBatch, name: &str) -> usize {
    batch.schema().index_of(name).expect("column not found")
}

/// Units for ra/dec columns: "deg" (degrees) or "rad" (radians).
fn is_radians(units: &str) -> bool {
    units.eq_ignore_ascii_case("rad")
}

/// Extract ra/dec as f64 in degrees. Supports Float32 and Float64; converts from radians if needed.
fn ra_dec_degrees(
    batch: &RecordBatch,
    ra_idx: usize,
    dec_idx: usize,
    from_radians: bool,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
    let n = batch.num_rows();
    let ra_col = batch.column(ra_idx);
    let dec_col = batch.column(dec_idx);
    let to_deg = if from_radians { RAD_TO_DEG } else { 1.0 };
    let ra: Vec<f64> = match ra_col.data_type() {
        DataType::Float64 => ra_col
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("ra: Float64 downcast")?
            .iter()
            .map(|v| v.unwrap_or(0.0) * to_deg)
            .collect(),
        DataType::Float32 => ra_col
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or("ra: Float32 downcast")?
            .iter()
            .map(|v| f64::from(v.unwrap_or(0.0)) * to_deg)
            .collect(),
        _ => return Err("ra column must be Float32 or Float64".into()),
    };
    let dec: Vec<f64> = match dec_col.data_type() {
        DataType::Float64 => dec_col
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("dec: Float64 downcast")?
            .iter()
            .map(|v| v.unwrap_or(0.0) * to_deg)
            .collect(),
        DataType::Float32 => dec_col
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or("dec: Float32 downcast")?
            .iter()
            .map(|v| f64::from(v.unwrap_or(0.0)) * to_deg)
            .collect(),
        _ => return Err("dec column must be Float32 or Float64".into()),
    };
    debug_assert_eq!(ra.len(), n);
    debug_assert_eq!(dec.len(), n);
    Ok((ra, dec))
}

/// List shard Parquet paths in directory (shard_0000.parquet, ...), sorted. Returns (paths, n_shards).
fn list_shard_paths(shard_dir: &Path) -> Result<(Vec<std::path::PathBuf>, usize), Box<dyn std::error::Error + Send + Sync>> {
    let mut paths: Vec<_> = std::fs::read_dir(shard_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("shard_") && n.ends_with(".parquet"))
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    let n = paths.len();
    if n == 0 {
        return Err("pre-partitioned B directory has no shard_*.parquet files".into());
    }
    Ok((paths, n))
}

/// Load B rows from pre-partitioned shards for the given set of pixel IDs.
/// Shard index = pixel_id % n_shards. Returns (id_b, ra_deg, dec_deg) for each row.
fn load_b_from_shards(
    shard_dir: &Path,
    pixels_wanted: &HashSet<u64>,
    n_shards: usize,
    from_radians: bool,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let (paths, _) = list_shard_paths(shard_dir)?;
    let shard_indices: HashSet<usize> = pixels_wanted.iter().map(|&p| (p % n_shards as u64) as usize).collect();
    let mut out: Vec<(IdVal, f64, f64)> = Vec::new();
    for &s in &shard_indices {
        if s >= paths.len() {
            continue;
        }
        let file = File::open(&paths[s])?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let pix_col = batch.column(column_index(&batch, "pixel_id"));
            let _id_col = batch.column(column_index(&batch, "id_b"));
            let ra_col = batch.column(column_index(&batch, "ra"));
            let dec_col = batch.column(column_index(&batch, "dec"));
            let to_deg = if from_radians { RAD_TO_DEG } else { 1.0 };
            for row in 0..batch.num_rows() {
                let pix_val = match pix_col.data_type() {
                    DataType::UInt64 => pix_col.as_any().downcast_ref::<UInt64Array>().unwrap().value(row),
                    DataType::Int64 => pix_col.as_any().downcast_ref::<Int64Array>().unwrap().value(row) as u64,
                    _ => continue,
                };
                if !pixels_wanted.contains(&pix_val) {
                    continue;
                }
                let id_val = get_id_value(&batch, "id_b", column_index(&batch, "id_b"), row);
                let ra = match ra_col.data_type() {
                    DataType::Float64 => ra_col.as_any().downcast_ref::<Float64Array>().unwrap().value(row) * to_deg,
                    DataType::Float32 => f64::from(ra_col.as_any().downcast_ref::<Float32Array>().unwrap().value(row)) * to_deg,
                    _ => continue,
                };
                let dec = match dec_col.data_type() {
                    DataType::Float64 => dec_col.as_any().downcast_ref::<Float64Array>().unwrap().value(row) * to_deg,
                    DataType::Float32 => f64::from(dec_col.as_any().downcast_ref::<Float32Array>().unwrap().value(row)) * to_deg,
                    _ => continue,
                };
                out.push((id_val, ra, dec));
            }
        }
    }
    Ok(out)
}

/// Compute pixels in chunk A plus all 8 neighbors (halo).
fn pixels_in_chunk_with_neighbors(ra_deg: &[f64], dec_deg: &[f64], depth: u8) -> HashSet<u64> {
    let layer = get(depth);
    let mut pixels = HashSet::new();
    for i in 0..ra_deg.len() {
        let lon = ra_deg[i] * DEG_TO_RAD;
        let lat = dec_deg[i] * DEG_TO_RAD;
        let center = layer.hash(lon, lat);
        pixels.insert(center);
        let mut neighbours = vec![center];
        nested::append_bulk_neighbours(depth, center, &mut neighbours);
        for p in neighbours {
            pixels.insert(p);
        }
    }
    pixels
}

/// Progress callback type: (chunk_ix, total_or_none, rows_a_read, matches_count).
pub type ProgressCallback = Option<Box<dyn Fn(usize, Option<usize>, u64, u64) + Send>>;

/// Cross-match two Parquet catalogs: stream A in chunks, build HEALPix index per chunk,
/// stream B (or load B from pre-partitioned shards), haversine filter, write matches.
/// When catalog_b is a directory with shard_*.parquet, B is loaded by pixel from shards.
/// Returns CrossMatchStats. If n_nearest is Some(n), keeps only n smallest-separation matches per id_a.
pub fn cross_match_impl(
    catalog_a: &Path,
    catalog_b: &Path,
    output_path: &Path,
    radius_arcsec: f64,
    depth: u8,
    batch_size_a: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_a: Option<&str>,
    id_col_b: Option<&str>,
    ra_dec_units: &str,
    n_nearest: Option<u32>,
    progress_callback: ProgressCallback,
) -> Result<CrossMatchStats, Box<dyn std::error::Error + Send + Sync>> {
    let from_radians = is_radians(ra_dec_units);
    let layer = get(depth);
    let t0 = std::time::Instant::now();

    let b_is_prepartitioned = catalog_b.is_dir();
    let (n_shards, shard_dir) = if b_is_prepartitioned {
        let (_paths, n) = list_shard_paths(catalog_b)?;
        (n, Some(catalog_b.to_path_buf()))
    } else {
        (0, None)
    };

    let file_a = File::open(catalog_a)?;
    let mut reader_a = ParquetRecordBatchReaderBuilder::try_new(file_a)?
        .with_batch_size(batch_size_a)
        .build()?;

    let mut id_a_name: Option<String> = None;
    let mut id_b_name: Option<String> = None;
    let mut first_dt_a: Option<DataType> = None;
    let mut first_dt_b: Option<DataType> = None;
    let mut out_schema: Option<Arc<Schema>> = None;
    let mut writer: Option<ArrowWriter<BufWriter<File>>> = None;
    let mut rows_a_read: u64 = 0;
    let mut rows_b_read: u64 = 0;
    let mut matches_count: u64 = 0;
    let mut chunks_processed: usize = 0;

    // If B is a single file, count rows once for stats (we'll re-read it per chunk).
    if !b_is_prepartitioned {
        let file_b = File::open(catalog_b)?;
        let meta = ParquetRecordBatchReaderBuilder::try_new(file_b)?
            .metadata()
            .clone();
        rows_b_read = meta
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as u64)
            .sum();
    }

    while let Some(batch_a) = reader_a.next() {
        let batch_a = batch_a?;
        if batch_a.num_rows() == 0 {
            continue;
        }

        chunks_processed += 1;
        let n_a = batch_a.num_rows();
        rows_a_read += n_a as u64;

        let id_a_col = id_column(&batch_a, ra_col, dec_col, id_col_a);
        if id_a_name.is_none() {
            id_a_name = Some(id_a_col.clone());
            first_dt_a = Some(id_type(&batch_a, &id_a_col));
        }

        let ra_idx = column_index(&batch_a, ra_col);
        let dec_idx = column_index(&batch_a, dec_col);
        let id_a_idx = column_index(&batch_a, &id_a_col);

        let (ra_deg_a, dec_deg_a) = ra_dec_degrees(&batch_a, ra_idx, dec_idx, from_radians)?;

        // Build index: pixel -> [(id, ra_deg, dec_deg), ...]
        let mut index: HashMap<u64, Vec<(IdVal, f64, f64)>> = HashMap::new();
        for row in 0..batch_a.num_rows() {
            let ra_deg = ra_deg_a[row];
            let dec_deg = dec_deg_a[row];
            let lon = ra_deg * DEG_TO_RAD;
            let lat = dec_deg * DEG_TO_RAD;
            let pix = layer.hash(lon, lat);
            let id_val = get_id_value(&batch_a, &id_a_col, id_a_idx, row);
            index
                .entry(pix)
                .or_default()
                .push((id_val, ra_deg, dec_deg));
        }

        let mut matches_id_a: Vec<IdVal> = Vec::new();
        let mut matches_id_b: Vec<IdVal> = Vec::new();
        let mut matches_sep: Vec<f64> = Vec::new();

        if b_is_prepartitioned {
            if id_b_name.is_none() {
                id_b_name = Some("id_b".to_string());
            }
            let pixels_wanted = pixels_in_chunk_with_neighbors(&ra_deg_a, &dec_deg_a, depth);
            let b_rows = load_b_from_shards(
                shard_dir.as_ref().unwrap(),
                &pixels_wanted,
                n_shards,
                from_radians,
            )?;
            for (id_b_val, ra_b_deg, dec_b_deg) in &b_rows {
                let center_pix = layer.hash(ra_b_deg * DEG_TO_RAD, dec_b_deg * DEG_TO_RAD);
                let mut pixels_to_look = vec![center_pix];
                nested::append_bulk_neighbours(depth, center_pix, &mut pixels_to_look);
                for pix in &pixels_to_look {
                    if let Some(candidates) = index.get(pix) {
                        for (id_a, ra_a, dec_a) in candidates {
                            let sep = haversine_arcsec(*ra_a, *dec_a, *ra_b_deg, *dec_b_deg);
                            if sep <= radius_arcsec {
                                matches_id_a.push(id_a.clone());
                                matches_id_b.push(id_b_val.clone());
                                matches_sep.push(sep);
                            }
                        }
                    }
                }
            }
            if !matches_id_a.is_empty() && out_schema.is_none() {
                let (col_a, col_b) = build_id_columns(&matches_id_a, &matches_id_b);
                let dt_a = col_a.data_type().clone();
                let dt_b = col_b.data_type().clone();
                let id_b_col_name = id_b_name.as_deref().unwrap_or("id_b");
                out_schema = Some(Arc::new(Schema::new(vec![
                    Field::new(id_a_col.as_str(), dt_a, false),
                    Field::new(id_b_col_name, dt_b, false),
                    Field::new("separation_arcsec", DataType::Float64, false),
                ])));
                let out_file = File::create(output_path)?;
                writer = Some(ArrowWriter::try_new(
                    BufWriter::new(out_file),
                    out_schema.clone().unwrap(),
                    Some(WriterProperties::builder().build()),
                )?);
            }
        } else {
            // Single-file B: re-read from start for this A chunk
            let file_b_again = File::open(catalog_b)?;
            let mut reader_b_inner =
                ParquetRecordBatchReaderBuilder::try_new(file_b_again)?
                    .with_batch_size(batch_size_b)
                    .build()?;

            while let Some(batch_b) = reader_b_inner.next() {
                let batch_b = batch_b?;
                if batch_b.num_rows() == 0 {
                    continue;
                }

                if id_b_name.is_none() {
                    let name = id_column(&batch_b, ra_col, dec_col, id_col_b);
                    id_b_name = Some(name.clone());
                    first_dt_b = Some(id_type(&batch_b, &name));
                }
                let id_b_col = id_b_name.as_ref().unwrap();
                let ra_b_idx = column_index(&batch_b, ra_col);
                let dec_b_idx = column_index(&batch_b, dec_col);
                let id_b_idx = column_index(&batch_b, id_b_col);

                let (ra_deg_b, dec_deg_b) =
                    ra_dec_degrees(&batch_b, ra_b_idx, dec_b_idx, from_radians)?;

                // Parallelize over B rows: each row produces a vec of (id_a, id_b, sep)
                let n_b = batch_b.num_rows();
                let index_ref = &index;
                let radius = radius_arcsec;
                let depth_local = depth;
                let id_b_vals: Vec<IdVal> = (0..n_b)
                    .map(|row| get_id_value(&batch_b, id_b_col, id_b_idx, row))
                    .collect();

                let per_row: Vec<Vec<(IdVal, IdVal, f64)>> = (0..n_b)
                    .into_par_iter()
                    .map(|row| {
                        let ra_b_deg = ra_deg_b[row];
                        let dec_b_deg = dec_deg_b[row];
                        let lon_b = ra_b_deg * DEG_TO_RAD;
                        let lat_b = dec_b_deg * DEG_TO_RAD;
                        let center_pix = layer.hash(lon_b, lat_b);
                        let mut pixels_to_look = vec![center_pix];
                        nested::append_bulk_neighbours(depth_local, center_pix, &mut pixels_to_look);
                        let id_b_val = id_b_vals[row].clone();
                        let mut row_matches = Vec::new();
                        for pix in &pixels_to_look {
                            if let Some(candidates) = index_ref.get(pix) {
                                for (id_a, ra_a, dec_a) in candidates {
                                    let sep = haversine_arcsec(*ra_a, *dec_a, ra_b_deg, dec_b_deg);
                                    if sep <= radius {
                                        row_matches.push((id_a.clone(), id_b_val.clone(), sep));
                                    }
                                }
                            }
                        }
                        row_matches
                    })
                    .collect();

                for row_matches in per_row {
                    for (id_a, id_b, sep) in row_matches {
                        matches_id_a.push(id_a);
                        matches_id_b.push(id_b);
                        matches_sep.push(sep);
                    }
                }

                if !matches_id_a.is_empty() && out_schema.is_none() {
                    let (col_a, col_b) = build_id_columns(&matches_id_a, &matches_id_b);
                    let dt_a = col_a.data_type().clone();
                    let dt_b = col_b.data_type().clone();
                    let id_b_col = id_b_name.as_ref().unwrap();
                    out_schema = Some(Arc::new(Schema::new(vec![
                        Field::new(id_a_col.as_str(), dt_a, false),
                        Field::new(id_b_col, dt_b, false),
                        Field::new("separation_arcsec", DataType::Float64, false),
                    ])));
                    let out_file = File::create(output_path)?;
                    writer = Some(ArrowWriter::try_new(
                        BufWriter::new(out_file),
                        out_schema.clone().unwrap(),
                        Some(WriterProperties::builder().build()),
                    )?);
                }
            }
        }

        matches_count += matches_id_a.len() as u64;

        if let Some(ref progress) = progress_callback {
            progress(chunks_processed, None, rows_a_read, matches_count);
        }

        if let Some(ref mut w) = writer {
            let (col_a, col_b) = build_id_columns(&matches_id_a, &matches_id_b);
            let schema = out_schema.as_ref().unwrap();
            let col_a = coerce_id_column_to_type(col_a, schema.field(0).data_type())?;
            let col_b = coerce_id_column_to_type(col_b, schema.field(1).data_type())?;
            let sep_arr = Arc::new(Float64Array::from(matches_sep));
            let id_b_col = id_b_name.as_deref().unwrap_or("id_b");
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new(id_a_col.as_str(), schema.field(0).data_type().clone(), false),
                    Field::new(id_b_col, schema.field(1).data_type().clone(), false),
                    Field::new("separation_arcsec", DataType::Float64, false),
                ])),
                vec![col_a, col_b, sep_arr],
            )?;
            w.write(&batch)?;
        }
    }

    if b_is_prepartitioned {
        let (paths, n) = list_shard_paths(shard_dir.as_ref().unwrap())?;
        rows_b_read = 0;
        for path in paths.iter().take(n) {
            let f = File::open(path)?;
            let meta = ParquetRecordBatchReaderBuilder::try_new(f)?.metadata().clone();
            rows_b_read += meta
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows() as u64)
                .sum::<u64>();
        }
    }

    if let Some(w) = writer.take() {
        w.close()?;
    } else {
        let id_a_col = id_a_name.as_deref().unwrap_or("id_a");
        let id_b_col = id_b_name.as_deref().unwrap_or("id_b");
        let dt_a = first_dt_a.clone().unwrap_or(DataType::Int64);
        let dt_b = first_dt_b.clone().unwrap_or(DataType::Utf8);
        let empty_schema = Arc::new(Schema::new(vec![
            Field::new(id_a_col, dt_a, false),
            Field::new(id_b_col, dt_b, false),
            Field::new("separation_arcsec", DataType::Float64, false),
        ]));
        let empty_batch = RecordBatch::new_empty(empty_schema.clone());
        let out_file = File::create(output_path)?;
        let mut w = ArrowWriter::try_new(
            BufWriter::new(out_file),
            empty_schema,
            Some(WriterProperties::builder().build()),
        )?;
        w.write(&empty_batch)?;
        w.close()?;
    }

    let id_a_col = id_a_name.as_deref().unwrap_or("id_a").to_string();
    let id_b_col = id_b_name.as_deref().unwrap_or("id_b").to_string();

    if let Some(n) = n_nearest {
        apply_n_nearest(output_path, &id_a_col, &id_b_col, n)?;
        let file_out = File::open(output_path)?;
        let meta = ParquetRecordBatchReaderBuilder::try_new(file_out)?.metadata().clone();
        matches_count = meta
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as u64)
            .sum();
    }

    let time_seconds = t0.elapsed().as_secs_f64();
    Ok(CrossMatchStats {
        output_path: output_path.to_string_lossy().to_string(),
        rows_a_read,
        rows_b_read,
        matches_count,
        chunks_processed,
        time_seconds,
    })
}

/// Entry for the per-id_a max-heap: keep n_nearest smallest by separation.
#[derive(Clone)]
struct NearestEntry {
    sep: f64,
    id_b: IdVal,
}

impl PartialEq for NearestEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sep.total_cmp(&other.sep) == Ordering::Equal
    }
}

impl Eq for NearestEntry {}

impl PartialOrd for NearestEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NearestEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sep.total_cmp(&other.sep)
    }
}

/// Keep only n_nearest smallest-separation matches per id_a; overwrite output file.
/// Streams batches so memory is O(distinct id_a × n_nearest), not O(total rows).
fn apply_n_nearest(
    output_path: &Path,
    id_a_col: &str,
    id_b_col: &str,
    n_nearest: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(output_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let schema = reader.schema().clone();
    let id_a_idx = schema.index_of(id_a_col)?;
    let id_b_idx = schema.index_of(id_b_col)?;
    let sep_idx = schema.index_of("separation_arcsec")?;
    let n_nearest_usize = n_nearest as usize;

    // Per id_a: max-heap of (sep, id_b); we keep at most n_nearest smallest.
    let mut by_a: HashMap<IdVal, BinaryHeap<NearestEntry>> = HashMap::new();

    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let sep_arr = batch
            .column(sep_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("separation_arcsec not Float64")?;

        for i in 0..batch.num_rows() {
            let id_a = get_id_value(&batch, id_a_col, id_a_idx, i);
            let id_b = get_id_value(&batch, id_b_col, id_b_idx, i);
            let sep = sep_arr.value(i);
            let heap = by_a.entry(id_a).or_default();
            let entry = NearestEntry {
                sep,
                id_b,
            };
            if heap.len() < n_nearest_usize {
                heap.push(entry);
            } else if sep < heap.peek().map(|e| e.sep).unwrap_or(f64::MAX) {
                heap.pop();
                heap.push(entry);
            }
        }
    }

    let mut out_id_a: Vec<IdVal> = Vec::new();
    let mut out_id_b: Vec<IdVal> = Vec::new();
    let mut out_sep: Vec<f64> = Vec::new();
    for (id_a, heap) in by_a {
        let mut list: Vec<_> = heap.into_iter().collect();
        list.sort_by(|a, b| a.sep.total_cmp(&b.sep));
        for e in list.into_iter().take(n_nearest_usize) {
            out_id_a.push(id_a.clone());
            out_id_b.push(e.id_b);
            out_sep.push(e.sep);
        }
    }

    let (col_a, col_b) = build_id_columns(&out_id_a, &out_id_b);
    let out_sep_arr = Arc::new(Float64Array::from(out_sep));
    let out_batch = RecordBatch::try_new(
        schema.clone(),
        vec![col_a, col_b, out_sep_arr],
    )?;
    let out_file = File::create(output_path)?;
    let buf = BufWriter::new(out_file);
    let mut w = ArrowWriter::try_new(
        buf,
        schema,
        Some(WriterProperties::builder().build()),
    )?;
    w.write(&out_batch)?;
    w.close()?;
    Ok(())
}

/// Coerce an ID column array to the target schema type so output batches match the writer schema.
fn coerce_id_column_to_type(
    col: Arc<dyn Array>,
    target: &DataType,
) -> Result<Arc<dyn Array>, Box<dyn std::error::Error + Send + Sync>> {
    if col.data_type() == target {
        return Ok(col);
    }
    cast(col.as_ref(), target).map_err(|e| e.into())
}

fn build_id_columns(
    id_a: &[IdVal],
    id_b: &[IdVal],
) -> (Arc<dyn Array>, Arc<dyn Array>) {
    let has_int_a = id_a.iter().any(|v| matches!(v, IdVal::I64(_)));
    let has_int_b = id_b.iter().any(|v| matches!(v, IdVal::I64(_)));
    let col_a: Arc<dyn Array> = if has_int_a {
        Arc::new(Int64Array::from_iter(id_a.iter().map(|v| {
            match v {
                IdVal::I64(x) => Some(*x),
                IdVal::Str(s) => s.parse().ok(),
            }
        })))
    } else {
        Arc::new(StringArray::from_iter(id_a.iter().map(|v| match v {
            IdVal::I64(x) => Some(x.to_string()),
            IdVal::Str(s) => Some(s.clone()),
        })))
    };
    let col_b: Arc<dyn Array> = if has_int_b {
        Arc::new(Int64Array::from_iter(id_b.iter().map(|v| {
            match v {
                IdVal::I64(x) => Some(*x),
                IdVal::Str(s) => s.parse().ok(),
            }
        })))
    } else {
        Arc::new(StringArray::from_iter(id_b.iter().map(|v| match v {
            IdVal::I64(x) => Some(x.to_string()),
            IdVal::Str(s) => Some(s.clone()),
        })))
    };
    (col_a, col_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;

    fn batch_with_int64_id() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("source_id", DataType::Int64, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
        ]);
        let id = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        let ra = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0]));
        let dec = Arc::new(Float64Array::from(vec![-5.0, 0.0, 5.0]));
        RecordBatch::try_new(
            Arc::new(schema),
            vec![id, ra, dec],
        )
        .unwrap()
    }

    fn batch_with_str_id() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("object_id", DataType::Utf8, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
        ]);
        let id = Arc::new(StringArray::from(vec!["A1", "A2", "A3"]));
        let ra = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0]));
        let dec = Arc::new(Float64Array::from(vec![-5.0, 0.0, 5.0]));
        RecordBatch::try_new(
            Arc::new(schema),
            vec![id, ra, dec],
        )
        .unwrap()
    }

    fn batch_with_int32_id() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
        ]);
        let id = Arc::new(Int32Array::from(vec![100, 200]));
        let ra = Arc::new(Float64Array::from(vec![0.0, 1.0]));
        let dec = Arc::new(Float64Array::from(vec![0.0, 1.0]));
        RecordBatch::try_new(Arc::new(schema), vec![id, ra, dec]).unwrap()
    }

    #[test]
    fn test_haversine_arcsec_same_point() {
        let sep = haversine_arcsec(10.0, -5.0, 10.0, -5.0);
        assert!((sep - 0.0).abs() < 1e-10, "same point should be 0 arcsec, got {}", sep);
    }

    #[test]
    fn test_haversine_arcsec_known_pair() {
        let sep = haversine_arcsec(0.0, 0.0, 0.0, 1.0);
        let expected_arcsec = 3600.0;
        assert!(
            (sep - expected_arcsec).abs() < 1.0,
            "1 deg separation ~3600 arcsec, got {}",
            sep
        );
    }

    #[test]
    fn test_id_column_inferred_first_non_ra_dec() {
        let batch = batch_with_int64_id();
        let name = id_column(&batch, "ra", "dec", None);
        assert_eq!(name, "source_id");
    }

    #[test]
    fn test_column_index() {
        let batch = batch_with_int64_id();
        assert_eq!(column_index(&batch, "ra"), 1);
        assert_eq!(column_index(&batch, "source_id"), 0);
    }

    #[test]
    fn test_build_id_columns_int64_and_str() {
        let id_a = vec![IdVal::I64(1), IdVal::I64(2)];
        let id_b = vec![IdVal::Str("B1".into()), IdVal::Str("B2".into())];
        let (col_a, col_b) = build_id_columns(&id_a, &id_b);
        assert_eq!(col_a.data_type(), &DataType::Int64);
        assert!(matches!(col_b.data_type(), DataType::Utf8 | DataType::LargeUtf8));
        assert_eq!(col_a.len(), 2);
        assert_eq!(col_b.len(), 2);
    }

    #[test]
    fn test_id_column_str_batch() {
        let batch = batch_with_str_id();
        let name = id_column(&batch, "ra", "dec", None);
        assert_eq!(name, "object_id");
    }

    #[test]
    fn test_id_type_int32_batch() {
        let batch = batch_with_int32_id();
        let dt = id_type(&batch, "id");
        assert_eq!(dt, DataType::Int32);
    }
}
