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

#[cfg(all(target_os = "macos", feature = "macos_readahead"))]
use std::os::unix::io::AsRawFd;

use tempfile::TempDir;

use arrow::array::{
    Array, ArrayAccessor, DictionaryArray, Float32Array, Float64Array, Int64Array, StringArray,
    UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use cdshealpix::nested::{self, get};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

/// Larger write buffer for temp shard files (fewer syscalls).
const SHARD_WRITE_BUFFER_BYTES: usize = 256 * 1024; // 256 KiB
/// Batch size when reading shard Parquet (larger = fewer decode cycles and larger I/O chunks).
const SHARD_READ_BATCH_ROWS: usize = 131_072; // 128k

/// On macOS, enable read-ahead on the file descriptor so sequential Parquet reads can be faster.
#[cfg(all(target_os = "macos", feature = "macos_readahead"))]
fn set_readahead(file: &File) {
    // F_RDAHEAD = 64 on Darwin (bsd/sys/fcntl.h)
    const F_RDAHEAD: libc::c_int = 64;
    let fd = file.as_raw_fd();
    unsafe {
        libc::fcntl(fd, F_RDAHEAD, 1);
    }
}

#[cfg(not(all(target_os = "macos", feature = "macos_readahead")))]
fn set_readahead(_file: &File) {}

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

/// Columnar B catalog slice: join hot path touches only ra_b/dec_b; id_b fetched when emitting a match.
#[derive(Clone)]
pub struct BColumns {
    pub id_b: Vec<IdVal>,
    pub ra_b: Vec<f64>,
    pub dec_b: Vec<f64>,
}

impl BColumns {
    #[inline]
    pub fn len(&self) -> usize {
        self.id_b.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_b.is_empty()
    }
}

impl From<Vec<(IdVal, f64, f64)>> for BColumns {
    fn from(rows: Vec<(IdVal, f64, f64)>) -> Self {
        let n = rows.len();
        let mut id_b = Vec::with_capacity(n);
        let mut ra_b = Vec::with_capacity(n);
        let mut dec_b = Vec::with_capacity(n);
        for (id, ra, dec) in rows {
            id_b.push(id);
            ra_b.push(ra);
            dec_b.push(dec);
        }
        BColumns { id_b, ra_b, dec_b }
    }
}

/// Haversine angular distance in radians (inputs in radians).
/// Standard formula: a = sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2); θ = 2 asin(√a).
#[inline(always)]
fn haversine_rad(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat * 0.5).sin().mul_add(
        (dlat * 0.5).sin(),
        lat1.cos() * lat2.cos() * (dlon * 0.5).sin() * (dlon * 0.5).sin(),
    );
    let c = 2.0 * (a.sqrt().min(1.0).asin());
    c
}

/// Haversine angular distance in arcsec (inputs in degrees).
#[inline(always)]
fn haversine_arcsec(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let lon1 = ra1_deg * DEG_TO_RAD;
    let lat1 = dec1_deg * DEG_TO_RAD;
    let lon2 = ra2_deg * DEG_TO_RAD;
    let lat2 = dec2_deg * DEG_TO_RAD;
    haversine_rad(lon1, lat1, lon2, lat2) * RAD_TO_ARCSEC
}

/// Compute 4 haversine distances (arcsec) in one batch. Same (ra2_deg, dec2_deg) for all four.
/// Writes to `out[0..4]`. Enables compiler autovectorization and better cache use.
#[inline(always)]
fn haversine_arcsec_4(
    ra1_deg: &[f64; 4],
    dec1_deg: &[f64; 4],
    ra2_deg: f64,
    dec2_deg: f64,
    out: &mut [f64; 4],
) {
    let lon2 = ra2_deg * DEG_TO_RAD;
    let lat2 = dec2_deg * DEG_TO_RAD;
    let cos_lat2 = lat2.cos();
    for i in 0..4 {
        let lon1 = ra1_deg[i] * DEG_TO_RAD;
        let lat1 = dec1_deg[i] * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let a = (dlat * 0.5).sin().mul_add(
            (dlat * 0.5).sin(),
            lat1.cos() * cos_lat2 * (dlon * 0.5).sin() * (dlon * 0.5).sin(),
        );
        out[i] = 2.0 * (a.sqrt().min(1.0).asin()) * RAD_TO_ARCSEC;
    }
}

/// Compute 8 haversine distances (arcsec) in one batch. Same (ra2_deg, dec2_deg) for all eight.
#[inline(always)]
fn haversine_arcsec_8(
    ra1_deg: &[f64; 8],
    dec1_deg: &[f64; 8],
    ra2_deg: f64,
    dec2_deg: f64,
    out: &mut [f64; 8],
) {
    let lon2 = ra2_deg * DEG_TO_RAD;
    let lat2 = dec2_deg * DEG_TO_RAD;
    let cos_lat2 = lat2.cos();
    for i in 0..8 {
        let lon1 = ra1_deg[i] * DEG_TO_RAD;
        let lat1 = dec1_deg[i] * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let a = (dlat * 0.5).sin().mul_add(
            (dlat * 0.5).sin(),
            lat1.cos() * cos_lat2 * (dlon * 0.5).sin() * (dlon * 0.5).sin(),
        );
        out[i] = 2.0 * (a.sqrt().min(1.0).asin()) * RAD_TO_ARCSEC;
    }
}

/// Cheap reject: true if (ra_a, dec_a) vs (ra_b, dec_b) is definitely outside radius_deg.
#[inline(always)]
fn cheap_reject_deg(ra_a_deg: f64, dec_a_deg: f64, ra_b_deg: f64, dec_b_deg: f64, radius_deg: f64) -> bool {
    if (dec_a_deg - dec_b_deg).abs() > radius_deg {
        return true;
    }
    let cos_dec = (dec_a_deg * DEG_TO_RAD).cos().abs().max((dec_b_deg * DEG_TO_RAD).cos().abs());
    if cos_dec < 1e-10 {
        return false;
    }
    (ra_a_deg - ra_b_deg).abs() * cos_dec > radius_deg
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
    id_type_from_schema(batch.schema().as_ref(), col_name)
}

/// Infer ID column type from schema (for use when no batch is available).
fn id_type_from_schema(schema: &Schema, col_name: &str) -> DataType {
    let field = schema
        .field_with_name(col_name)
        .expect("id column missing");
    match field.data_type() {
        DataType::Dictionary(_, value_type) => (**value_type).clone(),
        other => other.clone(),
    }
}

/// Resolve ID column name from schema (when no batch is available).
fn id_column_from_schema(
    schema: &Schema,
    ra_col: &str,
    dec_col: &str,
    id_col: Option<&str>,
) -> String {
    if let Some(c) = id_col {
        return c.to_string();
    }
    for i in 0..schema.fields().len() {
        let name = schema.field(i).name();
        if name.to_lowercase() != ra_col && name.to_lowercase() != dec_col {
            return name.clone();
        }
    }
    "index".to_string()
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
/// Uses a large batch size to reduce decode overhead; fast path for common shard schema (UInt64/Float64/Int64).
fn load_one_shard(
    path: &Path,
    pixels_wanted: &HashSet<u64>,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    set_readahead(&file);
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(SHARD_READ_BATCH_ROWS)
        .build()?;
    let mut out: Vec<(IdVal, f64, f64)> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }
        let pix_col = batch.column(column_index(&batch, "pixel_id"));
        let ra_col = batch.column(column_index(&batch, "ra"));
        let dec_col = batch.column(column_index(&batch, "dec"));
        let id_b_idx = column_index(&batch, "id_b");

        // Fast path: shard schema is always (pixel_id UInt64, id_b ?, ra Float64, dec Float64) from partition_b_to_temp.
        if let (Some(pix_arr), Some(ra_arr), Some(dec_arr)) = (
            pix_col.as_any().downcast_ref::<UInt64Array>(),
            ra_col.as_any().downcast_ref::<Float64Array>(),
            dec_col.as_any().downcast_ref::<Float64Array>(),
        ) {
            let ra_slice = ra_arr.values();
            let dec_slice = dec_arr.values();
            out.reserve(n);
            let id_col = batch.column(id_b_idx);
            if let Some(id_i64) = id_col.as_any().downcast_ref::<Int64Array>() {
                for row in 0..n {
                    if !pixels_wanted.contains(&pix_arr.value(row)) {
                        continue;
                    }
                    out.push((IdVal::I64(id_i64.value(row)), ra_slice[row], dec_slice[row]));
                }
            } else {
                for row in 0..n {
                    if !pixels_wanted.contains(&pix_arr.value(row)) {
                        continue;
                    }
                    out.push((
                        get_id_value(&batch, "id_b", id_b_idx, row),
                        ra_slice[row],
                        dec_slice[row],
                    ));
                }
            }
            continue;
        }

        // Fallback: other column types (e.g. Float32 ra/dec, Dictionary id_b).
        out.reserve(n);
        for row in 0..n {
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

/// Compute pixels in chunk A plus all 8 neighbors (halo). Parallel over rows; merge with
/// fold/reduce for better parallelism than single-threaded collect into HashSet.
fn pixels_in_chunk_with_neighbors(ra_deg: &[f64], dec_deg: &[f64], depth: u8) -> HashSet<u64> {
    let layer = get(depth);
    (0..ra_deg.len())
        .into_par_iter()
        .map(|i| {
            let lon = ra_deg[i] * DEG_TO_RAD;
            let lat = dec_deg[i] * DEG_TO_RAD;
            let center = layer.hash(lon, lat);
            let mut neighbours = Vec::with_capacity(9);
            neighbours.push(center);
            nested::append_bulk_neighbours(depth, center, &mut neighbours);
            neighbours
        })
        .fold_with(HashSet::new(), |mut s, neighbours| {
            s.extend(neighbours);
            s
        })
        .reduce_with(|mut a, b| {
            a.extend(b);
            a
        })
        .unwrap_or_else(HashSet::new)
}

/// One pass over chunk A: build HEALPix index (pixel -> row indices) and pixel set (centers + neighbors).
/// Hashes each (ra, dec) once instead of separate pixels then index passes.
fn pixels_and_index(
    ra_deg: &[f64],
    dec_deg: &[f64],
    depth: u8,
) -> (HashMap<u64, Vec<usize>>, HashSet<u64>) {
    let layer = get(depth);
    (0..ra_deg.len())
        .into_par_iter()
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            let center = layer.hash(lon, lat);
            let mut neighbours = Vec::with_capacity(9);
            neighbours.push(center);
            nested::append_bulk_neighbours(depth, center, &mut neighbours);
            (center, row, neighbours)
        })
        .fold_with(
            (HashMap::<u64, Vec<usize>>::new(), HashSet::<u64>::new()),
            |(mut idx, mut pixels), (center, row, neighbours)| {
                idx.entry(center).or_default().push(row);
                for p in neighbours {
                    pixels.insert(p);
                }
                (idx, pixels)
            },
        )
        .reduce_with(|(mut a_idx, mut a_pix), (b_idx, b_pix)| {
            for (p, rows) in b_idx {
                a_idx.entry(p).or_default().extend(rows);
            }
            a_pix.extend(b_pix);
            (a_idx, a_pix)
        })
        .unwrap_or_else(|| (HashMap::new(), HashSet::new()))
}

/// Build HEALPix index only (pixel -> row indices). One hash per row, no neighbor set.
fn index_only(
    ra_deg: &[f64],
    dec_deg: &[f64],
    depth: u8,
) -> HashMap<u64, Vec<usize>> {
    let layer = get(depth);
    (0..ra_deg.len())
        .into_par_iter()
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            (layer.hash(lon, lat), row)
        })
        .fold_with(
            HashMap::<u64, Vec<usize>>::new(),
            |mut m, (pix, row)| {
                m.entry(pix).or_default().push(row);
                m
            },
        )
        .reduce_with(|mut a, b| {
            for (pix, v) in b {
                a.entry(pix).or_default().extend(v);
            }
            a
        })
        .unwrap_or_else(HashMap::new)
}

/// Progress callback type: (chunk_ix, total_or_none, rows_a_read, matches_count).
/// Return true to continue, false to cancel (e.g. on Ctrl+C); engine stops at chunk boundary.
pub type ProgressCallback = Option<Box<dyn Fn(usize, Option<usize>, u64, u64) -> bool + Send>>;

#[inline]
fn verbose_log(msg: &str) {
    if env::var("PLEIADES_VERBOSE").is_ok() {
        eprintln!("[pleiades] {}", msg);
    }
}

fn verbose_log_timed(phase: &str, elapsed_secs: f64, extra: &str) {
    if env::var("PLEIADES_VERBOSE").is_ok() {
        eprintln!("[pleiades] {}: {:.3}s {}", phase, elapsed_secs, extra);
    }
}

/// Process one B batch into (shard_ix, pixel_id, id_val, ra, dec) for parallel partition.
fn partition_batch_to_row_results(
    batch: &RecordBatch,
    depth: u8,
    n_shards: usize,
    id_b_name: &str,
    ra_col: &str,
    dec_col: &str,
    from_radians: bool,
) -> Result<Vec<(usize, u64, IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let layer = get(depth);
    let ra_idx = column_index(batch, ra_col);
    let dec_idx = column_index(batch, dec_col);
    let id_b_idx = column_index(batch, id_b_name);
    let (ra_deg, dec_deg) = ra_dec_degrees(batch, ra_idx, dec_idx, from_radians)?;
    let n = batch.num_rows();
    let row_results: Vec<(usize, u64, IdVal, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            let pixel_id = layer.hash(lon, lat);
            let shard_ix = (pixel_id % n_shards as u64) as usize;
            let id_val = get_id_value(batch, id_b_name, id_b_idx, row);
            (shard_ix, pixel_id, id_val, ra_deg[row], dec_deg[row])
        })
        .collect();
    Ok(row_results)
}

/// Partition catalog B (single file) into HEALPix shards in a temp directory.
/// Returns (temp_dir_guard, shard_dir_path, rows_written, original_id_b_column_name).
/// Caller must keep the guard alive. Batches are processed in parallel (pixel assignment).
fn partition_b_to_temp(
    catalog_b: &Path,
    depth: u8,
    n_shards: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_b: Option<&str>,
    from_radians: bool,
) -> Result<(TempDir, PathBuf, u64, String), Box<dyn std::error::Error + Send + Sync>> {
    let temp_dir = tempfile::tempdir().map_err(|e| format!("temp dir: {}", e))?;
    let shard_dir = temp_dir.path().to_path_buf();

    let file_b = File::open(catalog_b)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file_b)?;
    let arrow_schema = builder.schema().clone();
    let mut reader = builder.with_batch_size(batch_size_b).build()?;

    let temp_shard_props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .build();

    let first_batch = match reader.next() {
        Some(Ok(b)) => b,
        Some(Err(e)) => return Err(e.into()),
        None => {
            let id_b_name_owned =
                id_column_from_schema(arrow_schema.as_ref(), ra_col, dec_col, id_col_b);
            let id_b_type = id_type_from_schema(arrow_schema.as_ref(), &id_b_name_owned);
            let shard_schema = Arc::new(Schema::new(vec![
                Field::new("pixel_id", DataType::UInt64, false),
                Field::new("id_b", id_b_type, false),
                Field::new("ra", DataType::Float64, false),
                Field::new("dec", DataType::Float64, false),
            ]));
            for s in 0..n_shards {
                let path = shard_dir.join(format!("shard_{:04}.parquet", s));
                let f = File::create(&path).map_err(|e| format!("create shard {}: {}", s, e))?;
                let mut w = ArrowWriter::try_new(
                    BufWriter::with_capacity(SHARD_WRITE_BUFFER_BYTES, f),
                    Arc::clone(&shard_schema),
                    Some(temp_shard_props.clone()),
                )?;
                w.write(&RecordBatch::new_empty(Arc::clone(&shard_schema)))?;
                w.close()?;
            }
            return Ok((temp_dir, shard_dir, 0, id_b_name_owned));
        }
    };

    let id_b_name = id_column(&first_batch, ra_col, dec_col, id_col_b);
    let id_b_name_owned = id_b_name.clone();

    if first_batch.num_rows() == 0 {
        let id_b_type = id_type(&first_batch, &id_b_name);
        let shard_schema = Arc::new(Schema::new(vec![
            Field::new("pixel_id", DataType::UInt64, false),
            Field::new("id_b", id_b_type, false),
            Field::new("ra", DataType::Float64, false),
            Field::new("dec", DataType::Float64, false),
        ]));
        for s in 0..n_shards {
            let path = shard_dir.join(format!("shard_{:04}.parquet", s));
            let f = File::create(&path).map_err(|e| format!("create shard {}: {}", s, e))?;
            let mut w = ArrowWriter::try_new(
                BufWriter::with_capacity(SHARD_WRITE_BUFFER_BYTES, f),
                Arc::clone(&shard_schema),
                Some(temp_shard_props.clone()),
            )?;
            w.write(&RecordBatch::new_empty(Arc::clone(&shard_schema)))?;
            w.close()?;
        }
        return Ok((temp_dir, shard_dir, 0, id_b_name_owned));
    }
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
                BufWriter::with_capacity(SHARD_WRITE_BUFFER_BYTES, f),
                Arc::clone(&shard_schema),
                Some(temp_shard_props.clone()),
            )?))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    let mut rows_written: u64 = 0;
    let mut buffers: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards).map(|_| Vec::new()).collect();
    /// Larger flush reduces Parquet write() calls and RecordBatch builds during partition.
    const FLUSH_AT: usize = 131_072;

    // Collect all batches then process in parallel (pixel assignment per batch).
    let mut batches: Vec<RecordBatch> = vec![first_batch];
    for batch in reader {
        let batch = batch?;
        if batch.num_rows() > 0 {
            batches.push(batch);
        }
    }
    let batch_results: Vec<Vec<(usize, u64, IdVal, f64, f64)>> = batches
        .par_iter()
        .map(|b| partition_batch_to_row_results(b, depth, n_shards, &id_b_name, ra_col, dec_col, from_radians))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    for row_results in batch_results {
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

    Ok((temp_dir, shard_dir, rows_written, id_b_name_owned))
}

/// Partition catalog B (single file) into in-memory shard buffers (no disk writes).
/// Returns (shard_buffers, original_id_b_column_name, rows_read). Use with load_b_from_memory.
fn partition_b_to_memory(
    catalog_b: &Path,
    depth: u8,
    n_shards: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_b: Option<&str>,
    from_radians: bool,
) -> Result<(Vec<Vec<(u64, IdVal, f64, f64)>>, String, u64), Box<dyn std::error::Error + Send + Sync>> {
    let layer = get(depth);
    let file_b = File::open(catalog_b)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file_b)?;
    let arrow_schema = builder.schema().clone();
    let mut reader = builder.with_batch_size(batch_size_b).build()?;

    let first_batch = match reader.next() {
        Some(Ok(b)) => b,
        Some(Err(e)) => return Err(e.into()),
        None => {
            let id_b_name_owned =
                id_column_from_schema(arrow_schema.as_ref(), ra_col, dec_col, id_col_b);
            let empty: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards).map(|_| Vec::new()).collect();
            return Ok((empty, id_b_name_owned, 0));
        }
    };

    let id_b_name = id_column(&first_batch, ra_col, dec_col, id_col_b);
    let id_b_name_owned = id_b_name.clone();
    let mut buffers: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut rows_read: u64 = 0;

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
        rows_read += n as u64;
        Ok(())
    };

    process_batch(&first_batch)?;
    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        process_batch(&batch)?;
    }
    Ok((buffers, id_b_name_owned, rows_read))
}

/// Load B rows from in-memory shard buffers for the given set of pixel IDs (no I/O).
fn load_b_from_memory(
    shard_buffers: &[Vec<(u64, IdVal, f64, f64)>],
    pixels_wanted: &HashSet<u64>,
    n_shards: usize,
) -> Vec<(IdVal, f64, f64)> {
    let shard_indices: Vec<usize> = pixels_wanted
        .iter()
        .map(|&p| (p % n_shards as u64) as usize)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let out: Vec<(IdVal, f64, f64)> = shard_indices
        .par_iter()
        .filter(|&&s| s < shard_buffers.len())
        .flat_map(|&s| {
            shard_buffers[s]
                .iter()
                .filter(|(pix, _, _, _)| pixels_wanted.contains(pix))
                .map(|(_, id_val, ra, dec)| (id_val.clone(), *ra, *dec))
                .collect::<Vec<_>>()
        })
        .collect();
    out
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

/// Group B row indices by HEALPix pixel (one hash per row).
fn group_b_by_pixel(
    ra_b: &[f64],
    dec_b: &[f64],
    depth: u8,
) -> HashMap<u64, Vec<usize>> {
    let layer = get(depth);
    let mut by_pixel: HashMap<u64, Vec<usize>> = HashMap::new();
    for (b_row_ix, (&ra, &dec)) in ra_b.iter().zip(dec_b.iter()).enumerate() {
        let pix = layer.hash(ra * DEG_TO_RAD, dec * DEG_TO_RAD);
        by_pixel.entry(pix).or_default().push(b_row_ix);
    }
    by_pixel
}

/// CPU join: columnar B, group by pixel (reuse pixels_to_look per pixel), cheap reject, haversine in batches of 8.
/// Returns per-row matches (candidate_ix, b_row_ix, sep_arcsec).
fn run_cpu_join(
    b_cols: &BColumns,
    index_ref: &HashMap<u64, Vec<usize>>,
    ra_flat_ref: &[f64],
    dec_flat_ref: &[f64],
    radius_arcsec: f64,
    radius_deg: f64,
    depth_local: u8,
) -> Vec<Vec<(usize, usize, f64)>> {
    let ra_b = &b_cols.ra_b[..];
    let dec_b = &b_cols.dec_b[..];
    let by_pixel = group_b_by_pixel(ra_b, dec_b, depth_local);

    let per_pixel: Vec<Vec<(usize, Vec<(usize, usize, f64)>)>> = by_pixel
        .par_iter()
        .map(|(pixel, b_indices)| {
            let mut pixels_to_look = Vec::with_capacity(9);
            pixels_to_look.push(*pixel);
            nested::append_bulk_neighbours(depth_local, *pixel, &mut pixels_to_look);

            let mut out: Vec<(usize, Vec<(usize, usize, f64)>)> = Vec::with_capacity(b_indices.len());
            for &b_row_ix in b_indices {
                let ra_b_deg = ra_b[b_row_ix];
                let dec_b_deg = dec_b[b_row_ix];
                let mut row_matches = Vec::new();
                for pix in &pixels_to_look {
                    let Some(candidate_ixs) = index_ref.get(pix) else { continue };
                    // Batches of 8 (then 4, then remainder)
                    let chunks8 = candidate_ixs.chunks_exact(8);
                    let remainder8 = chunks8.remainder();
                    for chunk in chunks8 {
                        let mut ra1 = [0.0_f64; 8];
                        let mut dec1 = [0.0_f64; 8];
                        for (i, &ix) in chunk.iter().enumerate() {
                            ra1[i] = ra_flat_ref[ix];
                            dec1[i] = dec_flat_ref[ix];
                        }
                        let mut seps = [0.0_f64; 8];
                        haversine_arcsec_8(&ra1, &dec1, ra_b_deg, dec_b_deg, &mut seps);
                        for (i, &candidate_ix) in chunk.iter().enumerate() {
                            if seps[i] <= radius_arcsec {
                                row_matches.push((candidate_ix, b_row_ix, seps[i]));
                            }
                        }
                    }
                    let chunks4 = remainder8.chunks_exact(4);
                    let remainder = chunks4.remainder();
                    for chunk in chunks4 {
                        let mut ra1 = [0.0_f64; 4];
                        let mut dec1 = [0.0_f64; 4];
                        for (i, &ix) in chunk.iter().enumerate() {
                            ra1[i] = ra_flat_ref[ix];
                            dec1[i] = dec_flat_ref[ix];
                        }
                        let mut seps = [0.0_f64; 4];
                        haversine_arcsec_4(&ra1, &dec1, ra_b_deg, dec_b_deg, &mut seps);
                        for (i, &candidate_ix) in chunk.iter().enumerate() {
                            if seps[i] <= radius_arcsec {
                                row_matches.push((candidate_ix, b_row_ix, seps[i]));
                            }
                        }
                    }
                    for &candidate_ix in remainder {
                        let ra_a = ra_flat_ref[candidate_ix];
                        let dec_a = dec_flat_ref[candidate_ix];
                        if cheap_reject_deg(ra_a, dec_a, ra_b_deg, dec_b_deg, radius_deg) {
                            continue;
                        }
                        let sep = haversine_arcsec(ra_a, dec_a, ra_b_deg, dec_b_deg);
                        if sep <= radius_arcsec {
                            row_matches.push((candidate_ix, b_row_ix, sep));
                        }
                    }
                }
                out.push((b_row_ix, row_matches));
            }
            out
        })
        .collect();

    let mut row_matches: Vec<Vec<(usize, usize, f64)>> = (0..b_cols.len()).map(|_| Vec::new()).collect();
    for chunk in per_pixel {
        for (b_row_ix, m) in chunk {
            row_matches[b_row_ix].extend(m);
        }
    }
    row_matches
}

/// Cross-match two Parquet catalogs: stream A in chunks, build HEALPix index per chunk,
/// stream B (or load B from pre-partitioned shards), haversine filter, write matches.
/// When catalog_b is a directory with shard_*.parquet, B is loaded by pixel from shards.
/// When catalog_b is a file, B is partitioned in-Rust: to temp dir (one read + shard writes) or to memory (one read, no shard I/O).
/// If keep_b_in_memory is true and B is a file, B is partitioned into RAM so no shard files are read per chunk (faster when B fits in RAM).
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
    keep_b_in_memory: bool,
    progress_callback: ProgressCallback,
) -> Result<CrossMatchStats, Box<dyn std::error::Error + Send + Sync>> {
    let from_radians = is_radians(ra_dec_units);
    #[allow(unused_variables)]
    let layer = get(depth);
    let radius_deg = radius_arcsec / 3600.0;
    let t0 = std::time::Instant::now();
    let verbose = env::var("PLEIADES_VERBOSE").is_ok();

    if verbose {
        match std::thread::available_parallelism() {
            Ok(p) => verbose_log(&format!(
                "parallelism: {} threads (set RAYON_NUM_THREADS to override)",
                p.get()
            )),
            Err(_) => verbose_log("parallelism: unknown (set RAYON_NUM_THREADS if needed)"),
        }
        let n_rayon = (0..1)
            .into_par_iter()
            .map(|_| rayon::current_num_threads())
            .find_any(|_| true)
            .unwrap_or(1);
        verbose_log(&format!("Rayon pool: {} threads", n_rayon));
    }

    let b_is_prepartitioned = catalog_b.is_dir();
    let _temp_partition: Option<TempDir>;
    let mut rows_b_from_partition: Option<u64> = None;
    let mut id_b_name_from_partition: Option<String> = None;
    let shard_dir: Option<PathBuf>;
    let b_in_memory: Option<Arc<Vec<Vec<(u64, IdVal, f64, f64)>>>>;

    let n_shards = if b_is_prepartitioned {
        let (_paths, n) = list_shard_paths(catalog_b)?;
        verbose_log("using pre-partitioned B (directory)");
        shard_dir = Some(catalog_b.to_path_buf());
        b_in_memory = None;
        n
    } else if keep_b_in_memory {
        verbose_log("partitioning B in-Rust (in-memory, single read, no shard I/O)...");
        let t_part = std::time::Instant::now();
        let (buffers, id_b_orig, rows_b) = partition_b_to_memory(
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
        id_b_name_from_partition = Some(id_b_orig);
        shard_dir = None;
        b_in_memory = Some(Arc::new(buffers));
        _temp_partition = None;
        if verbose {
            verbose_log_timed("partition B (memory)", t_part.elapsed().as_secs_f64(), &format!("({} rows, {} shards)", rows_b, n_shards));
        }
        n_shards
    } else {
        verbose_log("partitioning B in-Rust (single read)...");
        let t_part = std::time::Instant::now();
        let (temp_guard, dir_path, rows_b, id_b_orig) = partition_b_to_temp(
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
        id_b_name_from_partition = Some(id_b_orig);
        _temp_partition = Some(temp_guard);
        shard_dir = Some(dir_path);
        b_in_memory = None;
        if verbose {
            verbose_log_timed("partition B", t_part.elapsed().as_secs_f64(), &format!("({} rows, {} shards)", rows_b, n_shards));
        }
        n_shards
    };

    let use_b_prefetch = shard_dir.is_some() && b_in_memory.is_none();
    let a_channel_cap = if use_b_prefetch { 2 } else { 1 };
    if verbose && use_b_prefetch {
        verbose_log("B prefetch: overlap index||load B and join||load B (two requests in flight)");
    }

    // Prefetch thread: read catalog A batches one ahead (or two when B prefetch) so I/O overlaps with compute.
    let (tx_a, rx_a) = mpsc::sync_channel::<Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>>>(a_channel_cap);
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

    type BLoadResult = Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>>;
    // Capacity 2: send current + next B request so index||load B and join||load B overlap.
    let (tx_b_request, rx_b_request) = mpsc::sync_channel::<Option<HashSet<u64>>>(2);
    let (tx_b_response, rx_b_response) = mpsc::sync_channel::<BLoadResult>(2);
    let b_prefetch_handle = if use_b_prefetch {
        let shard_dir_b = shard_dir.as_ref().unwrap().clone();
        Some(thread::spawn(move || {
            while let Ok(Some(pixels)) = rx_b_request.recv() {
                match load_b_from_shards(&shard_dir_b, &pixels, n_shards, from_radians) {
                    Ok(rows) => {
                        let _ = tx_b_response.send(Ok(rows));
                    }
                    Err(e) => {
                        let _ = tx_b_response.send(Err(e));
                    }
                }
            }
        }))
    } else {
        None
    };

    // Receive one A batch, skipping empty; None = end of stream.
    let recv_a = |rx: &mpsc::Receiver<Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>>>| -> Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>> {
        loop {
            match rx.recv() {
                Ok(Ok(Some(b))) if b.num_rows() > 0 => return Ok(Some(b)),
                Ok(Ok(Some(_))) => continue,
                Ok(Ok(None)) => return Ok(None),
                Ok(Err(e)) => return Err(e),
                Err(mpsc::RecvError) => return Ok(None),
            }
        }
    };

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
    let mut cancelled = false;

    let mut current_batch = recv_a(&rx_a)?;
    let mut next_batch = if use_b_prefetch { recv_a(&rx_a)? } else { None };
    let mut prefetched_b: Option<Vec<(IdVal, f64, f64)>> = None;
    // Reuse pixels computed for B prefetch as pixels_wanted next iteration (avoids re-computing).
    let mut next_pixels_wanted: Option<HashSet<u64>> = None;

    while let Some(batch_a) = current_batch.take() {
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

        let t_pixels_index = std::time::Instant::now();
        let (index, pixels_wanted) = if let Some(reuse) = next_pixels_wanted.take() {
            // Chunk 1+: reuse pixels from previous B-prefetch; only build index.
            let index = index_only(&ra_deg_a, &dec_deg_a, depth);
            (index, reuse)
        } else {
            // Chunk 0: single pass for both index and pixel set.
            pixels_and_index(&ra_deg_a, &dec_deg_a, depth)
        };

        // Send B load requests so prefetch runs in parallel: current chunk (when first), and next chunk.
        if use_b_prefetch {
            if prefetched_b.is_none() {
                let _ = tx_b_request.send(Some(pixels_wanted.clone()));
            }
            if let Some(ref next) = next_batch {
                let ra_idx_n = column_index(next, ra_col);
                let dec_idx_n = column_index(next, dec_col);
                let (ra_n, dec_n) = ra_dec_degrees(next, ra_idx_n, dec_idx_n, from_radians)?;
                let pixels_next = pixels_in_chunk_with_neighbors(&ra_n, &dec_n, depth);
                let _ = tx_b_request.send(Some(pixels_next.clone()));
                next_pixels_wanted = Some(pixels_next);
            }
        }
        if verbose {
            verbose_log_timed(
                "  pixels+index",
                t_pixels_index.elapsed().as_secs_f64(),
                &format!("({} pixels)", pixels_wanted.len()),
            );
        }

        // Build id_a_flat and ra/dec flat arrays for join (index already built above).
        let id_a_flat: Vec<IdVal> = (0..n_a)
            .into_par_iter()
            .map(|row| get_id_value(&batch_a, &id_a_col, id_a_idx, row))
            .collect();
        let ra_flat: Vec<f64> = ra_deg_a.to_vec();
        let dec_flat: Vec<f64> = dec_deg_a.to_vec();

        let mut matches_id_a: Vec<IdVal> = Vec::new();
        let mut matches_id_b: Vec<IdVal> = Vec::new();
        let mut matches_sep: Vec<f64> = Vec::new();

        if id_b_name.is_none() {
            id_b_name = Some(
                id_b_name_from_partition
                    .clone()
                    .unwrap_or_else(|| "id_b".to_string()),
            );
        }
        let t_load = std::time::Instant::now();
        let b_rows_vec = if use_b_prefetch {
            prefetched_b.take().unwrap_or_else(|| {
                rx_b_response
                    .recv()
                    .expect("recv B for current chunk")
                    .expect("load B")
            })
        } else if let Some(ref buf) = b_in_memory {
            load_b_from_memory(buf, &pixels_wanted, n_shards)
        } else {
            load_b_from_shards(
                shard_dir.as_ref().unwrap(),
                &pixels_wanted,
                n_shards,
                from_radians,
            )?
        };
        let b_cols = BColumns::from(b_rows_vec);
        let n_b_loaded = b_cols.len();
        if verbose {
            verbose_log_timed("  load B", t_load.elapsed().as_secs_f64(), &format!("({} rows)", n_b_loaded));
        }
        let t_join = std::time::Instant::now();
        let index_ref = &index;
        let ra_flat_ref = &ra_flat;
        let dec_flat_ref = &dec_flat;
        let radius = radius_arcsec;
        let depth_local = depth;

        #[cfg(feature = "wgpu")]
        {
            let use_gpu = env::var("PLEIADES_GPU").as_deref() == Ok("wgpu") && crate::gpu::gpu_available();
            if use_gpu {
                // Collect (a_ix, b_ix) candidate pairs from HEALPix index (no distance yet).
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (b_row_ix, (&ra_b_deg, &dec_b_deg)) in b_cols.ra_b.iter().zip(b_cols.dec_b.iter()).enumerate() {
                    let center_pix = layer.hash(ra_b_deg * DEG_TO_RAD, dec_b_deg * DEG_TO_RAD);
                    let mut pixels_to_look = Vec::with_capacity(9);
                    pixels_to_look.push(center_pix);
                    nested::append_bulk_neighbours(depth_local, center_pix, &mut pixels_to_look);
                    for pix in &pixels_to_look {
                        if let Some(candidate_ixs) = index_ref.get(pix) {
                            for &candidate_ix in candidate_ixs {
                                pairs.push((candidate_ix, b_row_ix));
                            }
                        }
                    }
                }
                // GPU is only faster when pair count is very high (amortizes upload/readback).
                // Below threshold use CPU to avoid ~10–20× slowdown from chunk sync overhead.
                let min_pairs_for_gpu: usize = env::var("PLEIADES_GPU_MIN_PAIRS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(80_000_000);
                let use_gpu_this_chunk = pairs.len() >= min_pairs_for_gpu;
                if use_gpu_this_chunk && !pairs.is_empty() {
                    let ra_a_f32: Vec<f32> = pairs
                        .iter()
                        .map(|&(a_ix, _)| ra_flat_ref[a_ix] as f32)
                        .collect();
                    let dec_a_f32: Vec<f32> = pairs
                        .iter()
                        .map(|&(a_ix, _)| dec_flat_ref[a_ix] as f32)
                        .collect();
                    let ra_b_f32: Vec<f32> = pairs
                        .iter()
                        .map(|&(_, b_ix)| b_cols.ra_b[b_ix] as f32)
                        .collect();
                    let dec_b_f32: Vec<f32> = pairs
                        .iter()
                        .map(|&(_, b_ix)| b_cols.dec_b[b_ix] as f32)
                        .collect();
                    let a_ix_u32: Vec<u32> = pairs.iter().map(|&(a, _)| a as u32).collect();
                    let b_ix_u32: Vec<u32> = pairs.iter().map(|&(_, b)| b as u32).collect();
                    match crate::gpu::haversine_pairs_gpu_compact(
                        &ra_a_f32,
                        &dec_a_f32,
                        &ra_b_f32,
                        &dec_b_f32,
                        &a_ix_u32,
                        &b_ix_u32,
                        radius as f32,
                    ) {
                        Ok(gpu_matches) => {
                            for (a_ix, b_ix, sep) in gpu_matches {
                                matches_id_a.push(id_a_flat[a_ix as usize].clone());
                                matches_id_b.push(b_cols.id_b[b_ix as usize].clone());
                                matches_sep.push(sep as f64);
                            }
                            if verbose {
                                verbose_log_timed("  join (GPU)", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
                            }
                        }
                        Err(_) => {
                            if verbose {
                                verbose_log("  join (GPU failed, CPU fallback)");
                            }
                            let per_row = run_cpu_join(
                                &b_cols,
                                index_ref,
                                ra_flat_ref,
                                dec_flat_ref,
                                radius,
                                radius_deg,
                                depth_local,
                            );
                            for row_matches in per_row {
                                for (candidate_ix, b_row_ix, sep) in row_matches {
                                    matches_id_a.push(id_a_flat[candidate_ix].clone());
                                    matches_id_b.push(b_cols.id_b[b_row_ix].clone());
                                    matches_sep.push(sep);
                                }
                            }
                            if verbose {
                                verbose_log_timed("  join (CPU fallback)", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
                            }
                        }
                    }
                } else if use_gpu_this_chunk && pairs.is_empty() && verbose {
                    verbose_log_timed("  join (GPU)", t_join.elapsed().as_secs_f64(), "(0 matches)");
                } else {
                    // pairs below PLEIADES_GPU_MIN_PAIRS: use CPU (faster due to GPU chunk/sync overhead).
                    let per_row = run_cpu_join(
                        &b_cols,
                        index_ref,
                        ra_flat_ref,
                        dec_flat_ref,
                        radius,
                        radius_deg,
                        depth_local,
                    );
                    matches_id_a.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                    matches_id_b.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                    matches_sep.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                    for row_matches in per_row {
                        for (candidate_ix, b_row_ix, sep) in row_matches {
                            matches_id_a.push(id_a_flat[candidate_ix].clone());
                            matches_id_b.push(b_cols.id_b[b_row_ix].clone());
                            matches_sep.push(sep);
                        }
                    }
                    if verbose {
                        verbose_log_timed(
                            "  join (CPU)",
                            t_join.elapsed().as_secs_f64(),
                            &format!("({} matches, {} pairs below GPU threshold)", matches_id_a.len(), pairs.len()),
                        );
                    }
                }
            } else {
                let per_row = run_cpu_join(
                    &b_cols,
                    index_ref,
                    ra_flat_ref,
                    dec_flat_ref,
                    radius,
                    radius_deg,
                    depth_local,
                );
                matches_id_a.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                matches_id_b.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                matches_sep.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
                for row_matches in per_row {
                    for (candidate_ix, b_row_ix, sep) in row_matches {
                        matches_id_a.push(id_a_flat[candidate_ix].clone());
                        matches_id_b.push(b_cols.id_b[b_row_ix].clone());
                        matches_sep.push(sep);
                    }
                }
                if verbose {
                    verbose_log_timed("  join (CPU)", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let per_row = run_cpu_join(
                &b_cols,
                index_ref,
                ra_flat_ref,
                dec_flat_ref,
                radius,
                radius_deg,
                depth_local,
            );
            matches_id_a.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
            matches_id_b.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
            matches_sep.reserve(per_row.iter().map(|v| v.len()).sum::<usize>());
            for row_matches in per_row {
                for (candidate_ix, b_row_ix, sep) in row_matches {
                    matches_id_a.push(id_a_flat[candidate_ix].clone());
                    matches_id_b.push(b_cols.id_b[b_row_ix].clone());
                    matches_sep.push(sep);
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        if verbose {
            verbose_log_timed("  join", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
        }
        // Per-chunk n_nearest: keep only best n per id_a before write (apply_n_nearest still merges across chunks).
        if let Some(n) = n_nearest {
            let (a, b, s) = merge_to_n_nearest(
                std::mem::take(&mut matches_id_a),
                std::mem::take(&mut matches_id_b),
                std::mem::take(&mut matches_sep),
                n,
            );
            matches_id_a = a;
            matches_id_b = b;
            matches_sep = s;
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
            if !progress(chunks_processed, None, rows_a_read, matches_count) {
                cancelled = true;
                break;
            }
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

        if use_b_prefetch {
            if next_batch.is_none() {
                break;
            }
            prefetched_b = Some(match rx_b_response.recv() {
                Ok(Ok(rows)) => rows,
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(e.into()),
            });
            current_batch = next_batch.take();
            next_batch = recv_a(&rx_a)?;
        } else {
            current_batch = recv_a(&rx_a)?;
        }
    }

    if use_b_prefetch {
        let _ = tx_b_request.send(None);
        if let Some(h) = b_prefetch_handle {
            let _ = h.join();
        }
    }

    if let Err(e) = reader_handle.join() {
        std::panic::resume_unwind(e);
    }

    if cancelled {
        return Err("cancelled by user".into());
    }

    if let Some(r) = rows_b_from_partition {
        rows_b_read = r;
    } else {
        let (paths, n) = list_shard_paths(shard_dir.as_ref().unwrap())?;
        rows_b_read = 0u64;
        for path in paths.iter().take(n) {
            let f = File::open(path)?;
            let meta = ParquetRecordBatchReaderBuilder::try_new(f)?
                .metadata()
            .clone();
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
        let meta = ParquetRecordBatchReaderBuilder::try_new(file_out)?
            .metadata()
        .clone();
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

/// Per-chunk: keep only n smallest-separation matches per id_a. Reduces match list before write.
fn merge_to_n_nearest(
    id_a: Vec<IdVal>,
    id_b: Vec<IdVal>,
    sep: Vec<f64>,
    n: u32,
) -> (Vec<IdVal>, Vec<IdVal>, Vec<f64>) {
    let n = n as usize;
    let mut by_a: HashMap<IdVal, Vec<(f64, IdVal)>> = HashMap::new();
    for ((a, b), s) in id_a.into_iter().zip(id_b).zip(sep) {
        by_a.entry(a).or_default().push((s, b));
    }
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut out_sep = Vec::new();
    for (a, mut list) in by_a {
        list.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        for (s, b) in list.into_iter().take(n) {
            out_a.push(a.clone());
            out_b.push(b);
            out_sep.push(s);
        }
    }
    (out_a, out_b, out_sep)
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
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
    .build()?;
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
