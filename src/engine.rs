//! Cross-match engine: stream Parquet A/B, HEALPix index, haversine join, write matches.
//! Supports pre-partitioned B (shard directory), n_nearest, parallelism, and progress callback.
//!
//! Uses `.expect()`/`.unwrap()` only on Arrow schema and column types that we control
//! (Parquet read with known ra/dec/id columns); invalid external data is rejected by the
//! Python validation layer before the engine is invoked.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use tempfile::TempDir;

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

/// Load one shard file, return rows whose pixel_id is in pixels_wanted. Shard ra/dec are in degrees.
fn load_one_shard(
    path: &Path,
    pixels_wanted: &HashSet<u64>,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut out: Vec<(IdVal, f64, f64)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let pix_col = batch.column(column_index(&batch, "pixel_id"));
        let ra_col = batch.column(column_index(&batch, "ra"));
        let dec_col = batch.column(column_index(&batch, "dec"));
        let id_b_idx = column_index(&batch, "id_b");
        for row in 0..batch.num_rows() {
            let pix_val = match pix_col.data_type() {
                DataType::UInt64 => pix_col.as_any().downcast_ref::<UInt64Array>().unwrap().value(row),
                DataType::Int64 => pix_col.as_any().downcast_ref::<Int64Array>().unwrap().value(row) as u64,
                _ => continue,
            };
            if !pixels_wanted.contains(&pix_val) {
                continue;
            }
            let id_val = get_id_value(&batch, "id_b", id_b_idx, row);
            let ra = match ra_col.data_type() {
                DataType::Float64 => ra_col.as_any().downcast_ref::<Float64Array>().unwrap().value(row),
                DataType::Float32 => f64::from(ra_col.as_any().downcast_ref::<Float32Array>().unwrap().value(row)),
                _ => continue,
            };
            let dec = match dec_col.data_type() {
                DataType::Float64 => dec_col.as_any().downcast_ref::<Float64Array>().unwrap().value(row),
                DataType::Float32 => f64::from(dec_col.as_any().downcast_ref::<Float32Array>().unwrap().value(row)),
                _ => continue,
            };
            out.push((id_val, ra, dec));
        }
    }
    Ok(out)
}

/// Load B rows from pre-partitioned shards for the given set of pixel IDs (shards loaded in parallel).
/// Shard index = pixel_id % n_shards. Returns (id_b, ra_deg, dec_deg) for each row.
/// Shard files always store ra/dec in degrees (written by partition_b_to_temp).
fn load_b_from_shards(
    shard_dir: &Path,
    pixels_wanted: &HashSet<u64>,
    n_shards: usize,
    _from_radians: bool,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let (paths, _) = list_shard_paths(shard_dir)?;
    let shard_indices: Vec<usize> = pixels_wanted
        .iter()
        .map(|&p| (p % n_shards as u64) as usize)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let vecs: Vec<Result<Vec<(IdVal, f64, f64)>, _>> = shard_indices
        .par_iter()
        .filter(|&&s| s < paths.len())
        .map(|&s| load_one_shard(&paths[s], pixels_wanted))
        .collect();
    let out: Vec<(IdVal, f64, f64)> = vecs
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
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

#[inline]
fn verbose_log(msg: &str) {
    if env::var("ASTROJOIN_VERBOSE").is_ok() {
        eprintln!("[astrojoin] {}", msg);
    }
}

fn verbose_log_timed(phase: &str, elapsed_secs: f64, extra: &str) {
    if env::var("ASTROJOIN_VERBOSE").is_ok() {
        eprintln!("[astrojoin] {}: {:.3}s {}", phase, elapsed_secs, extra);
    }
}

/// Partition catalog B (single file) into HEALPix shards in a temp directory.
/// Returns (temp_dir_guard, shard_dir_path, n_shards). Caller must keep the guard alive.
fn partition_b_to_temp(
    catalog_b: &Path,
    depth: u8,
    n_shards: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_b: Option<&str>,
    from_radians: bool,
) -> Result<(TempDir, PathBuf, u64), Box<dyn std::error::Error + Send + Sync>> {
    let layer = get(depth);
    let temp_dir = tempfile::tempdir().map_err(|e| format!("temp dir: {}", e))?;
    let shard_dir = temp_dir.path().to_path_buf();

    let file_b = File::open(catalog_b)?;
    let mut reader =
        ParquetRecordBatchReaderBuilder::try_new(file_b)?
            .with_batch_size(batch_size_b)
            .build()?;

    let first_batch = match reader.next() {
        Some(Ok(b)) if b.num_rows() > 0 => b,
        _ => return Err("catalog B has no rows".into()),
    };

    let id_b_name = id_column(&first_batch, ra_col, dec_col, id_col_b);
    let id_b_type = id_type(&first_batch, &id_b_name);
    let shard_schema = Arc::new(Schema::new(vec![
        Field::new("pixel_id", DataType::UInt64, false),
        Field::new("id_b", id_b_type.clone(), false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]));

    let mut writers: Vec<Option<ArrowWriter<BufWriter<File>>>> = (0..n_shards)
        .map(|s| {
            let path = shard_dir.join(format!("shard_{:04}.parquet", s));
            let f = File::create(&path).map_err(|e| format!("create shard {}: {}", s, e))?;
            Ok(Some(ArrowWriter::try_new(
                BufWriter::new(f),
                Arc::clone(&shard_schema),
                Some(WriterProperties::builder().build()),
            )?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    let mut rows_written: u64 = 0;
    let mut buffers: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards).map(|_| Vec::new()).collect();
    const FLUSH_AT: usize = 16_384;

    let mut process_batch = |batch: &RecordBatch| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let ra_idx = column_index(batch, ra_col);
        let dec_idx = column_index(batch, dec_col);
        let id_b_idx = column_index(batch, &id_b_name);
        let (ra_deg, dec_deg) = ra_dec_degrees(batch, ra_idx, dec_idx, from_radians)?;
        let n = batch.num_rows();
        let row_results: Vec<(usize, u64, IdVal, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let lon = ra_deg[row] * DEG_TO_RAD;
                let lat = dec_deg[row] * DEG_TO_RAD;
                let pixel_id = layer.hash(lon, lat);
                let shard_ix = (pixel_id % n_shards as u64) as usize;
                let id_val = get_id_value(batch, &id_b_name, id_b_idx, row);
                (shard_ix, pixel_id, id_val, ra_deg[row], dec_deg[row])
            })
            .collect();
        for (shard_ix, pixel_id, id_val, ra, dec) in row_results {
            buffers[shard_ix].push((pixel_id, id_val, ra, dec));
        }
        for (s, buf) in buffers.iter_mut().enumerate() {
            while buf.len() >= FLUSH_AT {
                let chunk: Vec<(u64, IdVal, f64, f64)> = buf.drain(..FLUSH_AT).collect();
                let pix: Vec<u64> = chunk.iter().map(|r| r.0).collect();
                let ids: Vec<IdVal> = chunk.iter().map(|r| r.1.clone()).collect();
                let ras: Vec<f64> = chunk.iter().map(|r| r.2).collect();
                let decs: Vec<f64> = chunk.iter().map(|r| r.3).collect();
                let pix_arr = UInt64Array::from(pix);
                let id_arr = build_id_column_single(&ids);
                let ra_arr = Arc::new(Float64Array::from(ras));
                let dec_arr = Arc::new(Float64Array::from(decs));
                let batch_out = RecordBatch::try_new(
                    shard_schema.clone(),
                    vec![
                        Arc::new(pix_arr),
                        id_arr,
                        ra_arr,
                        dec_arr,
                    ],
                )?;
                writers[s].as_mut().unwrap().write(&batch_out)?;
                rows_written += FLUSH_AT as u64;
            }
        }
        Ok(())
    };

    // Process first batch (we already have it)
    process_batch(&first_batch)?;

    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        process_batch(&batch)?;
    }

    // Flush remaining buffers
    for (s, buf) in buffers.iter_mut().enumerate() {
        if buf.is_empty() {
            continue;
        }
        let pix: Vec<u64> = buf.iter().map(|r| r.0).collect();
        let ids: Vec<IdVal> = buf.iter().map(|r| r.1.clone()).collect();
        let ras: Vec<f64> = buf.iter().map(|r| r.2).collect();
        let decs: Vec<f64> = buf.iter().map(|r| r.3).collect();
        let pix_arr = UInt64Array::from(pix);
        let id_arr = build_id_column_single(&ids);
        let ra_arr = Arc::new(Float64Array::from(ras));
        let dec_arr = Arc::new(Float64Array::from(decs));
        let batch_out = RecordBatch::try_new(
            shard_schema.clone(),
            vec![
                Arc::new(pix_arr),
                id_arr,
                ra_arr,
                dec_arr,
            ],
        )?;
        writers[s].as_mut().unwrap().write(&batch_out)?;
        rows_written += buf.len() as u64;
    }

    for w in writers.iter_mut() {
        w.take().unwrap().close()?;
    }

    Ok((temp_dir, shard_dir, rows_written))
}

fn build_id_column_single(ids: &[IdVal]) -> Arc<dyn Array> {
    let has_int = ids.iter().any(|v| matches!(v, IdVal::I64(_)));
    if has_int {
        Arc::new(Int64Array::from_iter(ids.iter().map(|v| match v {
            IdVal::I64(x) => Some(*x),
            IdVal::Str(s) => s.parse().ok(),
        })))
    } else {
        Arc::new(StringArray::from_iter(ids.iter().map(|v| match v {
            IdVal::I64(x) => Some(x.to_string()),
            IdVal::Str(s) => Some(s.clone()),
        })))
    }
}

/// Cross-match two Parquet catalogs: stream A in chunks, build HEALPix index per chunk,
/// stream B (or load B from pre-partitioned shards), haversine filter, write matches.
/// When catalog_b is a directory with shard_*.parquet, B is loaded by pixel from shards.
/// When catalog_b is a file, B is partitioned in-Rust to a temp dir first (one read), then shard path is used.
/// Returns CrossMatchStats. If n_nearest is Some(n), keeps only n smallest-separation matches per id_a.
pub fn cross_match_impl(
    catalog_a: &Path,
    catalog_b: &Path,
    output_path: &Path,
    radius_arcsec: f64,
    depth: u8,
    batch_size_a: usize,
    batch_size_b: usize,
    n_shards: usize,
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
    let verbose = env::var("ASTROJOIN_VERBOSE").is_ok();

    let b_is_prepartitioned = catalog_b.is_dir();
    let _temp_partition: Option<TempDir>;
    let mut rows_b_from_partition: Option<u64> = None;
    let (n_shards, shard_dir) = if b_is_prepartitioned {
        let (_paths, n) = list_shard_paths(catalog_b)?;
        verbose_log("using pre-partitioned B (directory)");
        (n, Some(catalog_b.to_path_buf()))
    } else {
        verbose_log("partitioning B in-Rust (single read)...");
        let t_part = std::time::Instant::now();
        let (temp_guard, dir_path, rows_b) = partition_b_to_temp(
            catalog_b,
            depth,
            n_shards,
            batch_size_b,
            ra_col,
            dec_col,
            id_col_b,
            from_radians,
        )?;
        rows_b_from_partition = Some(rows_b);
        _temp_partition = Some(temp_guard);
        if verbose {
            verbose_log_timed("partition B", t_part.elapsed().as_secs_f64(), &format!("({} rows, {} shards)", rows_b, n_shards));
        }
        (n_shards, Some(dir_path))
    };

    // Prefetch thread: read catalog A batches one ahead so I/O overlaps with compute.
    let (tx_a, rx_a) = mpsc::sync_channel::<Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>>>(1);
    let path_a = catalog_a.to_path_buf();
    let batch_size_a_prefetch = batch_size_a;
    let reader_handle = thread::spawn(move || {
        let file_a = match File::open(&path_a) {
            Ok(f) => f,
            Err(e) => {
                let _ = tx_a.send(Err(e.into()));
                return;
            }
        };
        let reader_a = match ParquetRecordBatchReaderBuilder::try_new(file_a)
            .and_then(|b| b.with_batch_size(batch_size_a_prefetch).build())
        {
            Ok(r) => r,
            Err(e) => {
                let _ = tx_a.send(Err(e.into()));
                return;
            }
        };
        for batch in reader_a {
            match batch {
                Ok(b) => {
                    if tx_a.send(Ok(Some(b))).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx_a.send(Err(e.into()));
                    return;
                }
            }
        }
        let _ = tx_a.send(Ok(None));
    });

    let mut id_a_name: Option<String> = None;
    let mut id_b_name: Option<String> = None;
    let mut first_dt_a: Option<DataType> = None;
    let first_dt_b: Option<DataType> = None;
    let mut out_schema: Option<Arc<Schema>> = None;
    let mut writer: Option<ArrowWriter<BufWriter<File>>> = None;
    let mut rows_a_read: u64 = 0;
    let mut rows_b_read: u64;
    let mut matches_count: u64 = 0;
    let mut chunks_processed: usize = 0;

    while let Ok(msg) = rx_a.recv() {
        let batch_a = match msg {
            Ok(Some(b)) => b,
            Ok(None) => break,
            Err(e) => return Err(e),
        };
        if batch_a.num_rows() == 0 {
            continue;
        }

        chunks_processed += 1;
        let n_a = batch_a.num_rows();
        rows_a_read += n_a as u64;
        let t_chunk = std::time::Instant::now();
        if verbose {
            verbose_log(&format!("chunk {}: rows_a={}", chunks_processed, n_a));
        }

        let id_a_col = id_column(&batch_a, ra_col, dec_col, id_col_a);
        if id_a_name.is_none() {
            id_a_name = Some(id_a_col.clone());
            first_dt_a = Some(id_type(&batch_a, &id_a_col));
        }

        let ra_idx = column_index(&batch_a, ra_col);
        let dec_idx = column_index(&batch_a, dec_col);
        let id_a_idx = column_index(&batch_a, &id_a_col);

        let (ra_deg_a, dec_deg_a) = ra_dec_degrees(&batch_a, ra_idx, dec_idx, from_radians)?;

        let t_index = std::time::Instant::now();
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
        if verbose {
            verbose_log_timed("  index", t_index.elapsed().as_secs_f64(), "");
        }

        let mut matches_id_a: Vec<IdVal> = Vec::new();
        let mut matches_id_b: Vec<IdVal> = Vec::new();
        let mut matches_sep: Vec<f64> = Vec::new();

        if id_b_name.is_none() {
            id_b_name = Some("id_b".to_string());
        }
        let t_load = std::time::Instant::now();
        let pixels_wanted = pixels_in_chunk_with_neighbors(&ra_deg_a, &dec_deg_a, depth);
        let b_rows = load_b_from_shards(
            shard_dir.as_ref().unwrap(),
            &pixels_wanted,
            n_shards,
            from_radians,
        )?;
        let n_b_loaded = b_rows.len();
        if verbose {
            verbose_log_timed("  load B", t_load.elapsed().as_secs_f64(), &format!("({} rows)", n_b_loaded));
        }
        let t_join = std::time::Instant::now();
        let index_ref = &index;
        let radius = radius_arcsec;
        let depth_local = depth;
        let per_row: Vec<Vec<(IdVal, IdVal, f64)>> = b_rows
            .par_iter()
            .map(|(id_b_val, ra_b_deg, dec_b_deg)| {
                let center_pix = layer.hash(ra_b_deg * DEG_TO_RAD, dec_b_deg * DEG_TO_RAD);
                let mut pixels_to_look = vec![center_pix];
                nested::append_bulk_neighbours(depth_local, center_pix, &mut pixels_to_look);
                let mut row_matches = Vec::new();
                for pix in &pixels_to_look {
                    if let Some(candidates) = index_ref.get(pix) {
                        for (id_a, ra_a, dec_a) in candidates {
                            let sep = haversine_arcsec(*ra_a, *dec_a, *ra_b_deg, *dec_b_deg);
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
        if verbose {
            verbose_log_timed("  join", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
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

        matches_count += matches_id_a.len() as u64;

        if let Some(ref progress) = progress_callback {
            progress(chunks_processed, None, rows_a_read, matches_count);
        }

        if let Some(ref mut w) = writer {
            let t_write = std::time::Instant::now();
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
            if verbose {
                verbose_log_timed("  write", t_write.elapsed().as_secs_f64(), "");
                verbose_log_timed("  chunk total", t_chunk.elapsed().as_secs_f64(), "");
            }
        }
    }

    if let Err(e) = reader_handle.join() {
        std::panic::resume_unwind(e);
    }

    if let Some(r) = rows_b_from_partition {
        rows_b_read = r;
    } else {
        let (paths, n) = list_shard_paths(shard_dir.as_ref().unwrap())?;
        rows_b_read = 0u64;
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
    if verbose {
        verbose_log_timed("total", t0.elapsed().as_secs_f64(), &format!("({} chunks, {} matches)", chunks_processed, matches_count));
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
