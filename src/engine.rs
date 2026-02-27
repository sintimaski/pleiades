//! Cross-match engine: stream Parquet A/B, HEALPix index, haversine join, write matches.
//! Supports pre-partitioned B (shard directory), n_nearest, parallelism, and progress callback.
//!
//! Uses `.expect()`/`.unwrap()` only on Arrow schema and column types that we control
//! (Parquet read with known ra/dec/id columns); invalid external data is rejected by the
//! Python validation layer before the engine is invoked.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::env;

use arrayvec::ArrayVec;
use rustc_hash::{FxHashMap, FxHashSet};

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use std::fs::File;
use tempfile::TempDir;

use arrow::array::{
    Array, ArrayAccessor, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, StringArray, UInt64Array,
};
use arrow::compute::{cast, filter_record_batch};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use cdshealpix::nested::{self, get};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

#[cfg(feature = "rtree")]
use rstar::{primitives::GeomWithData, AABB, RTree};

#[cfg(feature = "simd")]
use crate::haversine_simd::{haversine_a_4_rad as haversine_a_4_rad_simd, haversine_a_8_rad as haversine_a_8_rad_simd};

/// Larger write buffer for temp shard files (fewer syscalls).
const SHARD_WRITE_BUFFER_BYTES: usize = 256 * 1024; // 256 KiB
/// Batch size when reading shard Parquet (larger = fewer decode cycles and larger I/O chunks).
const SHARD_READ_BATCH_ROWS: usize = 131_072; // 128k

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

/// Haversine intermediate term `a` only (no sqrt/asin). Used for cheap-reject: if a > a_max, skip.
/// a_max = sin²(radius_rad/2) for haversine threshold.
/// Uses sin_cos for lat1/lat2 to avoid redundant trig calls.
#[inline(always)]
fn haversine_a_rad(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let sin_hdlat = (dlat * 0.5).sin();
    let sin_hdlon = (dlon * 0.5).sin();
    let (_, cos_lat1) = lat1.sin_cos();
    let (_, cos_lat2) = lat2.sin_cos();
    sin_hdlat.mul_add(
        sin_hdlat,
        cos_lat1 * cos_lat2 * sin_hdlon * sin_hdlon,
    )
}

/// Convert haversine `a` to angular distance in arcsec (call only when a <= a_max).
#[inline(always)]
fn haversine_a_to_arcsec(a: f64) -> f64 {
    2.0 * (a.sqrt().min(1.0)).asin() * RAD_TO_ARCSEC
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

/// Unit vector (x,y,z) from ra/dec in degrees.
/// Uses sin_cos to compute both sin and cos in one call (cheaper than separate sin/cos).
#[inline(always)]
fn ra_dec_to_xyz_deg(ra_deg: f64, dec_deg: f64) -> (f64, f64, f64) {
    let lon = ra_deg * DEG_TO_RAD;
    let lat = dec_deg * DEG_TO_RAD;
    let (lon_sin, lon_cos) = lon.sin_cos();
    let (lat_sin, lat_cos) = lat.sin_cos();
    (lon_cos * lat_cos, lon_sin * lat_cos, lat_sin)
}

/// Angular separation in arcsec via dot product of unit vectors. Fast for small angles.
/// dot = cos(θ); θ = acos(dot); sep_arcsec = θ * RAD_TO_ARCSEC.
#[inline(always)]
#[allow(dead_code)]
fn dot_product_arcsec(
    x1: f64, y1: f64, z1: f64,
    x2: f64, y2: f64, z2: f64,
) -> f64 {
    let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
    dot.acos() * RAD_TO_ARCSEC
}

/// SIMD-friendly haversine for 4 lanes: branchless loop for autovectorization (sin/cos/asin stay scalar per lane).
#[inline(always)]
#[allow(dead_code)]
fn haversine_arcsec_4_rad(
    ra1_deg: &[f64; 4],
    dec1_deg: &[f64; 4],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 4],
) {
    for i in 0..4 {
        let lon1 = ra1_deg[i] * DEG_TO_RAD;
        let lat1 = dec1_deg[i] * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let half_dlat = dlat * 0.5;
        let half_dlon = dlon * 0.5;
        let sin_half_dlat = half_dlat.sin();
        let sin_half_dlon = half_dlon.sin();
        let a = sin_half_dlat.mul_add(
            sin_half_dlat,
            lat1.cos() * cos_lat2 * sin_half_dlon * sin_half_dlon,
        );
        let sqrt_a = a.sqrt().min(1.0);
        out[i] = 2.0 * sqrt_a.asin() * RAD_TO_ARCSEC;
    }
}

/// Haversine `a` term only for 4 lanes (cheap-reject: skip sqrt/asin when a > a_max).
/// When simd feature is on, uses AVX2/NEON via haversine_simd.
#[inline(always)]
fn haversine_a_4_rad(
    ra1_deg: &[f64; 4],
    dec1_deg: &[f64; 4],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 4],
) {
    #[cfg(feature = "simd")]
    {
        haversine_a_4_rad_simd(ra1_deg, dec1_deg, lon2, lat2, cos_lat2, out);
        return;
    }
    #[cfg(not(feature = "simd"))]
    for i in 0..4 {
        let lon1 = ra1_deg[i] * DEG_TO_RAD;
        let lat1 = dec1_deg[i] * DEG_TO_RAD;
        let dlat = lat2 - lat1;
        let dlon = lon2 - lon1;
        let half_dlat = dlat * 0.5;
        let half_dlon = dlon * 0.5;
        let sin_half_dlat = half_dlat.sin();
        let sin_half_dlon = half_dlon.sin();
        out[i] = sin_half_dlat.mul_add(
            sin_half_dlat,
            lat1.cos() * cos_lat2 * sin_half_dlon * sin_half_dlon,
        );
    }
}

/// SIMD-friendly haversine for 8 lanes: two 4-lane batches for cache and autovectorization.
#[inline(always)]
#[allow(dead_code)]
fn haversine_arcsec_8_rad(
    ra1_deg: &[f64; 8],
    dec1_deg: &[f64; 8],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 8],
) {
    let (ra1_4a, ra1_4b) = (
        [ra1_deg[0], ra1_deg[1], ra1_deg[2], ra1_deg[3]],
        [ra1_deg[4], ra1_deg[5], ra1_deg[6], ra1_deg[7]],
    );
    let (dec1_4a, dec1_4b) = (
        [dec1_deg[0], dec1_deg[1], dec1_deg[2], dec1_deg[3]],
        [dec1_deg[4], dec1_deg[5], dec1_deg[6], dec1_deg[7]],
    );
    let mut out_a = [0.0_f64; 4];
    let mut out_b = [0.0_f64; 4];
    haversine_arcsec_4_rad(&ra1_4a, &dec1_4a, lon2, lat2, cos_lat2, &mut out_a);
    haversine_arcsec_4_rad(&ra1_4b, &dec1_4b, lon2, lat2, cos_lat2, &mut out_b);
    out[0..4].copy_from_slice(&out_a);
    out[4..8].copy_from_slice(&out_b);
}

/// Haversine `a` term only for 8 lanes (cheap-reject: skip sqrt/asin when a > a_max).
/// When simd feature is on, delegates to haversine_simd (two 4-lane SIMD batches).
#[inline(always)]
fn haversine_a_8_rad(
    ra1_deg: &[f64; 8],
    dec1_deg: &[f64; 8],
    lon2: f64,
    lat2: f64,
    cos_lat2: f64,
    out: &mut [f64; 8],
) {
    #[cfg(feature = "simd")]
    {
        haversine_a_8_rad_simd(ra1_deg, dec1_deg, lon2, lat2, cos_lat2, out);
        return;
    }
    #[cfg(not(feature = "simd"))]
    {
        let (ra1_4a, ra1_4b) = (
            [ra1_deg[0], ra1_deg[1], ra1_deg[2], ra1_deg[3]],
            [ra1_deg[4], ra1_deg[5], ra1_deg[6], ra1_deg[7]],
        );
        let (dec1_4a, dec1_4b) = (
            [dec1_deg[0], dec1_deg[1], dec1_deg[2], dec1_deg[3]],
            [dec1_deg[4], dec1_deg[5], dec1_deg[6], dec1_deg[7]],
        );
        let mut out_a = [0.0_f64; 4];
        let mut out_b = [0.0_f64; 4];
        haversine_a_4_rad(&ra1_4a, &dec1_4a, lon2, lat2, cos_lat2, &mut out_a);
        haversine_a_4_rad(&ra1_4b, &dec1_4b, lon2, lat2, cos_lat2, &mut out_b);
        out[0..4].copy_from_slice(&out_a);
        out[4..8].copy_from_slice(&out_b);
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

/// Zero-copy or copied ra/dec in degrees. Use `.ra()` and `.dec()` to get `&[f64]`.
enum RaDecRef<'a> {
    /// Slice into Arrow Float64Array buffer (no copy).
    Slice(&'a [f64], &'a [f64]),
    /// Owned Vec when conversion (Float32, radians, or nulls) is needed.
    Owned(Vec<f64>, Vec<f64>),
}

impl<'a> RaDecRef<'a> {
    #[inline]
    fn ra(&self) -> &[f64] {
        match self {
            RaDecRef::Slice(ra, _) => ra,
            RaDecRef::Owned(ra, _) => ra.as_slice(),
        }
    }
    #[inline]
    fn dec(&self) -> &[f64] {
        match self {
            RaDecRef::Slice(_, dec) => dec,
            RaDecRef::Owned(_, dec) => dec.as_slice(),
        }
    }
}

/// Extract ra/dec as f64 in degrees. Zero-copy when Float64 + degrees + no nulls.
fn ra_dec_degrees<'a>(
    batch: &'a RecordBatch,
    ra_idx: usize,
    dec_idx: usize,
    from_radians: bool,
) -> Result<RaDecRef<'a>, Box<dyn std::error::Error + Send + Sync>> {
    let n = batch.num_rows();
    let ra_col = batch.column(ra_idx);
    let dec_col = batch.column(dec_idx);
    let to_deg = if from_radians { RAD_TO_DEG } else { 1.0 };

    // Zero-copy path: Float64, degrees (to_deg==1), no nulls.
    if to_deg == 1.0 {
        if let (Some(ra_arr), Some(dec_arr)) = (
            ra_col.as_any().downcast_ref::<Float64Array>(),
            dec_col.as_any().downcast_ref::<Float64Array>(),
        ) {
            if ra_arr.null_count() == 0 && dec_arr.null_count() == 0 {
                let ra = ra_arr.values();
                let dec = dec_arr.values();
                let ra_slice = &ra[..n];
                let dec_slice = &dec[..n];
                return Ok(RaDecRef::Slice(ra_slice, dec_slice));
            }
        }
    }

    // Copy path: Float32, radians, or nulls.
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
    Ok(RaDecRef::Owned(ra, dec))
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

/// Extract (id_b, ra, dec) from a shard RecordBatch for rows whose pixel_id is in pixels_wanted.
fn extract_shard_batch_rows(
    batch: &RecordBatch,
    pixels_wanted: &FxHashSet<u64>,
) -> Vec<(IdVal, f64, f64)> {
    let n = batch.num_rows();
    if n == 0 {
        return Vec::new();
    }
    let pix_col = batch.column(column_index(batch, "pixel_id"));
    let ra_col = batch.column(column_index(batch, "ra"));
    let dec_col = batch.column(column_index(batch, "dec"));
    let id_b_idx = column_index(batch, "id_b");
    // Reserve based on expected matches: when few pixels wanted, avoid over-allocating.
    let cap = if pixels_wanted.len() < 1000 {
        (pixels_wanted.len() * 256).min(n)
    } else {
        n
    };
    let mut out = Vec::with_capacity(cap);

    if let (Some(pix_arr), Some(ra_arr), Some(dec_arr)) = (
        pix_col.as_any().downcast_ref::<UInt64Array>(),
        ra_col.as_any().downcast_ref::<Float64Array>(),
        dec_col.as_any().downcast_ref::<Float64Array>(),
    ) {
        let ra_slice = ra_arr.values();
        let dec_slice = dec_arr.values();
        let id_col = batch.column(id_b_idx);
        if let Some(id_i64) = id_col.as_any().downcast_ref::<Int64Array>() {
            for row in 0..n {
                if pixels_wanted.contains(&pix_arr.value(row)) {
                    out.push((IdVal::I64(id_i64.value(row)), ra_slice[row], dec_slice[row]));
                }
            }
        } else {
            for row in 0..n {
                if pixels_wanted.contains(&pix_arr.value(row)) {
                    out.push((
                        get_id_value(batch, "id_b", id_b_idx, row),
                        ra_slice[row],
                        dec_slice[row],
                    ));
                }
            }
        }
        return out;
    }

    for row in 0..n {
        let pix_val = match pix_col.data_type() {
            DataType::UInt64 => pix_col.as_any().downcast_ref::<UInt64Array>().unwrap().value(row),
            DataType::Int64 => pix_col.as_any().downcast_ref::<Int64Array>().unwrap().value(row) as u64,
            _ => continue,
        };
        if !pixels_wanted.contains(&pix_val) {
            continue;
        }
        let id_val = get_id_value(batch, "id_b", id_b_idx, row);
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
    out
}

/// Load one shard file, return rows whose pixel_id is in pixels_wanted. Shard ra/dec are in degrees.
/// Uses parallel row-group decode when the shard has multiple row groups.
/// Pre-allocates output to avoid par_extend reallocs during collect.
fn load_one_shard(
    path: &Path,
    pixels_wanted: &FxHashSet<u64>,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let batches = crate::parquet_parallel::read_parquet_parallel(path, SHARD_READ_BATCH_ROWS)?;
    let per_batch: Vec<Vec<(IdVal, f64, f64)>> = batches
        .par_iter()
        .map(|b| extract_shard_batch_rows(b, pixels_wanted))
        .collect();
    let total: usize = per_batch.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for v in per_batch {
        out.extend(v);
    }
    Ok(out)
}

/// Load B rows from pre-partitioned shards for the given set of pixel IDs (shards loaded in parallel).
/// Shard index = pixel_id % n_shards. Returns (id_b, ra_deg, dec_deg) for each row.
/// Shard files always store ra/dec in degrees (written by partition_b_to_temp).
fn load_b_from_shards(
    shard_dir: &Path,
    pixels_wanted: &FxHashSet<u64>,
    n_shards: usize,
    _from_radians: bool,
) -> Result<Vec<(IdVal, f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
    let (paths, _) = list_shard_paths(shard_dir)?;
    let shard_indices: Vec<usize> = pixels_wanted
        .iter()
        .map(|&p| (p % n_shards as u64) as usize)
        .collect::<FxHashSet<_>>()
        .into_iter()
        .collect();
    let vecs: Vec<Result<Vec<(IdVal, f64, f64)>, _>> = shard_indices
        .par_iter()
        .filter(|&&s| s < paths.len())
        .map(|&s| load_one_shard(&paths[s], pixels_wanted))
        .collect();
    let vecs = vecs.into_iter().collect::<Result<Vec<_>, _>>()?;
    let total: usize = vecs.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total);
    for v in vecs {
        out.extend(v);
    }
    Ok(out)
}

/// Compute pixels in chunk A plus all 8 neighbors (halo). Parallel over rows; merge with
/// fold/reduce for better parallelism than single-threaded collect into HashSet.
/// Uses thread-local cache for neighbor lists to avoid repeated append_bulk_neighbours.
#[allow(dead_code)]
fn pixels_in_chunk_with_neighbors(ra_deg: &[f64], dec_deg: &[f64], depth: u8) -> FxHashSet<u64> {
    let layer = get(depth);
    (0..ra_deg.len())
        .into_par_iter()
        .map(|i| {
            let lon = ra_deg[i] * DEG_TO_RAD;
            let lat = dec_deg[i] * DEG_TO_RAD;
            let center = layer.hash(lon, lat);
            cached_neighbours(depth, center)
        })
        .fold_with(FxHashSet::default(), |mut s, neighbours| {
            s.extend(neighbours);
            s
        })
        .reduce_with(|mut a, b| {
            a.extend(b);
            a
        })
        .unwrap_or_else(FxHashSet::default)
}

/// Estimated unique pixels for capacity hint (reduces rehashing).
/// Add headroom to avoid resize during merge (HashMap rehash is expensive).
fn estimated_pixels(n_rows: usize) -> usize {
    let base = (n_rows / 4).min(300_000).max(4096);
    base + (base / 4) // 25% headroom
}

/// Max neighbors per pixel (center + 8).
const MAX_NEIGHBOURS: usize = 9;

/// Thread-local cache for HEALPix neighbor lists to avoid repeated append_bulk_neighbours calls.
fn cached_neighbours(depth: u8, center: u64) -> ArrayVec<u64, MAX_NEIGHBOURS> {
    thread_local! {
        static CACHE: RefCell<FxHashMap<(u8, u64), ArrayVec<u64, MAX_NEIGHBOURS>>> =
            RefCell::new(FxHashMap::default());
        static SCRATCH: RefCell<Vec<u64>> = RefCell::new(Vec::with_capacity(MAX_NEIGHBOURS));
    }
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if let Some(n) = cache.get(&(depth, center)) {
            return n.clone();
        }
        SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.clear();
            scratch.push(center);
            nested::append_bulk_neighbours(depth, center, &mut scratch);
            let arr: ArrayVec<u64, MAX_NEIGHBOURS> =
                scratch.drain(..).collect();
            cache.insert((depth, center), arr.clone());
            arr
        })
    })
}

/// Merge collected per-row results into (index, pixels). Used for profiling when PLEIADES_PROFILE=1.
fn merge_pixels_and_index(
    collected: Vec<(u64, usize, ArrayVec<u64, MAX_NEIGHBOURS>)>,
) -> (FxHashMap<u64, Vec<usize>>, FxHashSet<u64>) {
    let cap = estimated_pixels(collected.len());
    let base = (cap * 4) / 5; // cap = base + base/4, so base ≈ cap * 4/5
    let avg_rows_per_pixel = (collected.len() / base.max(1)).max(1);
    let mut idx: FxHashMap<u64, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(cap, Default::default());
    let mut pixels = FxHashSet::with_capacity_and_hasher(cap, Default::default());
    for (center, row, neighbours) in collected {
        idx.entry(center)
            .or_insert_with(|| Vec::with_capacity(avg_rows_per_pixel))
            .push(row);
        for p in neighbours {
            pixels.insert(p);
        }
    }
    (idx, pixels)
}

/// Merge collected (pixel, row) pairs into index. Used for profiling when PLEIADES_PROFILE=1.
fn merge_index_only(collected: Vec<(u64, usize)>) -> FxHashMap<u64, Vec<usize>> {
    let cap = estimated_pixels(collected.len());
    let base = (cap * 4) / 5;
    let avg_rows_per_pixel = (collected.len() / base.max(1)).max(1);
    let mut idx: FxHashMap<u64, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(cap, Default::default());
    for (pix, row) in collected {
        idx.entry(pix)
            .or_insert_with(|| Vec::with_capacity(avg_rows_per_pixel))
            .push(row);
    }
    idx
}

/// One pass over chunk A: build HEALPix index (pixel -> row indices) and pixel set (centers + neighbors).
/// Uses parallel map + single-threaded merge (avoids DashMap contention). Pre-allocates collected Vec.
fn pixels_and_index(
    ra_deg: &[f64],
    dec_deg: &[f64],
    depth: u8,
) -> (FxHashMap<u64, Vec<usize>>, FxHashSet<u64>) {
    let layer = get(depth);
    let profile = env::var("PLEIADES_PROFILE").is_ok();
    let t_map = std::time::Instant::now();
    let mut collected: Vec<(u64, usize, ArrayVec<u64, MAX_NEIGHBOURS>)> =
        Vec::with_capacity(ra_deg.len());
    (0..ra_deg.len())
        .into_par_iter()
        .with_min_len(4096)
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            let center = layer.hash(lon, lat);
            (center, row, cached_neighbours(depth, center))
        })
        .collect_into_vec(&mut collected);
    if profile {
        profile_log("pixels_and_index map", t_map.elapsed().as_secs_f64(), &format!("({} rows)", collected.len()));
    }
    let t_merge = std::time::Instant::now();
    let result = merge_pixels_and_index(collected);
    if profile {
        profile_log("pixels_and_index merge (HashSet/HashMap)", t_merge.elapsed().as_secs_f64(), &format!("({} pixels)", result.1.len()));
    }
    result
}

/// Build HEALPix index only (pixel -> row indices). One hash per row, no neighbor set.
/// Uses parallel map + single-threaded merge (avoids DashMap contention).
fn index_only(
    ra_deg: &[f64],
    dec_deg: &[f64],
    depth: u8,
) -> FxHashMap<u64, Vec<usize>> {
    let layer = get(depth);
    let profile = env::var("PLEIADES_PROFILE").is_ok();
    let t_map = std::time::Instant::now();
    let mut collected: Vec<(u64, usize)> = Vec::with_capacity(ra_deg.len());
    (0..ra_deg.len())
        .into_par_iter()
        .with_min_len(4096)
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            (layer.hash(lon, lat), row)
        })
        .collect_into_vec(&mut collected);
    if profile {
        profile_log("index_only map", t_map.elapsed().as_secs_f64(), &format!("({} rows)", collected.len()));
    }
    let t_merge = std::time::Instant::now();
    let result = merge_index_only(collected);
    if profile {
        profile_log("index_only merge (HashMap)", t_merge.elapsed().as_secs_f64(), &format!("({} pixels)", result.len()));
    }
    result
}

/// Progress callback type: (chunk_ix, total_or_none, rows_a_read, matches_count).
/// Return true to continue, false to cancel (e.g. on Ctrl+C); engine stops at chunk boundary.
pub type ProgressCallback = Option<Box<dyn Fn(usize, Option<usize>, u64, u64) -> bool + Send>>;

/// Match callback for streaming: receives (id_a, id_b, sep) batch per chunk.
/// Return true to continue, false to cancel. When Some, no output file is written.
pub type MatchCallback = Option<Box<dyn FnMut(Vec<IdVal>, Vec<IdVal>, Vec<f64>) -> bool + Send>>;

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

/// When PLEIADES_PROFILE=1, log a profiling line (phase + elapsed seconds + extra).
fn profile_log(phase: &str, elapsed_secs: f64, extra: &str) {
    if env::var("PLEIADES_PROFILE").is_ok() {
        eprintln!("[pleiades] profile {}: {:.3}s {}", phase, elapsed_secs, extra);
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
    let ra_dec = ra_dec_degrees(batch, ra_idx, dec_idx, from_radians)?;
    let ra_deg = ra_dec.ra();
    let dec_deg = ra_dec.dec();
    let n = batch.num_rows();
    let mut row_results: Vec<(usize, u64, IdVal, f64, f64)> = Vec::with_capacity(n);
    (0..n)
        .into_par_iter()
        .map(|row| {
            let lon = ra_deg[row] * DEG_TO_RAD;
            let lat = dec_deg[row] * DEG_TO_RAD;
            let pixel_id = layer.hash(lon, lat);
            let shard_ix = (pixel_id % n_shards as u64) as usize;
            let id_val = get_id_value(batch, id_b_name, id_b_idx, row);
            (shard_ix, pixel_id, id_val, ra_deg[row], dec_deg[row])
        })
        .collect_into_vec(&mut row_results);
    Ok(row_results)
}

/// Partition a catalog file into HEALPix shards in the given directory.
/// Writes shard_0000.parquet, ... to shard_dir. Returns (rows_written, id_column_name).
fn partition_file_to_shard_dir(
    catalog_path: &Path,
    shard_dir: &Path,
    depth: u8,
    n_shards: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_b: Option<&str>,
    from_radians: bool,
) -> Result<(u64, String), Box<dyn std::error::Error + Send + Sync>> {
    let temp_shard_props = WriterProperties::builder()
        .set_compression(Compression::UNCOMPRESSED)
        .build();

    let all_batches = crate::parquet_parallel::read_parquet_parallel(catalog_path, batch_size_b)?;
    let arrow_schema = match all_batches.first() {
        Some(b) => b.schema(),
        None => {
            let input_b = crate::parquet_mmap::ParquetInput::open(catalog_path)?;
            ParquetRecordBatchReaderBuilder::try_new(input_b)?.schema().clone()
        }
    };
    let first_batch = match all_batches.first() {
        Some(b) if b.num_rows() > 0 => b.clone(),
        _ => {
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
            return Ok((0, id_b_name_owned));
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
        return Ok((0, id_b_name_owned));
    }
    let id_b_type = id_type(&first_batch, &id_b_name);
    let shard_schema = Arc::new(Schema::new(vec![
        Field::new("pixel_id", DataType::UInt64, false),
        Field::new("id_b", id_b_type.clone(), false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]));

    let writers: Vec<Option<ArrowWriter<BufWriter<File>>>> = (0..n_shards)
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

    /// Larger flush reduces Parquet write() calls and RecordBatch builds during partition.
    const FLUSH_AT: usize = 131_072;

    // Batches from parallel decode (already includes first_batch).
    let batches: Vec<RecordBatch> = all_batches;
    let batch_results: Vec<Vec<(usize, u64, IdVal, f64, f64)>> = batches
        .par_iter()
        .map(|b| partition_batch_to_row_results(b, depth, n_shards, &id_b_name, ra_col, dec_col, from_radians))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>()?;

    // Group each batch's results by shard.
    let per_batch_per_shard: Vec<Vec<Vec<(u64, IdVal, f64, f64)>>> = batch_results
        .par_iter()
        .map(|rows| {
            let mut by_shard: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards).map(|_| Vec::new()).collect();
            for (shard_ix, pixel_id, id_val, ra, dec) in rows {
                by_shard[*shard_ix].push((*pixel_id, id_val.clone(), *ra, *dec));
            }
            by_shard
        })
        .collect();

    // Per-shard: merge all batches' rows and write in parallel.
    let shard_data: Vec<Vec<(u64, IdVal, f64, f64)>> = (0..n_shards)
        .into_par_iter()
        .map(|s| {
            per_batch_per_shard
                .iter()
                .flat_map(|batch| batch[s].iter().cloned())
                .collect()
        })
        .collect();

    let writer_data: Vec<(ArrowWriter<BufWriter<File>>, Vec<(u64, IdVal, f64, f64)>)> = writers
        .into_iter()
        .zip(shard_data)
        .map(|(opt, data)| (opt.unwrap(), data))
        .collect();

    let n_writers = writer_data.len();
    let mut write_results: Vec<Result<u64, Box<dyn std::error::Error + Send + Sync>>> =
        Vec::with_capacity(n_writers);
    writer_data
        .into_par_iter()
        .map(|(mut writer, data)| -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
            let mut written = 0u64;
            for chunk in data.chunks(FLUSH_AT) {
                let pix: Vec<u64> = chunk.iter().map(|r| r.0).collect();
                let ids: Vec<IdVal> = chunk.iter().map(|r| r.1.clone()).collect();
                let ras: Vec<f64> = chunk.iter().map(|r| r.2).collect();
                let decs: Vec<f64> = chunk.iter().map(|r| r.3).collect();
                let pix_arr = UInt64Array::from(pix);
                let id_arr = build_id_column_single(&ids);
                let ra_arr = Arc::new(Float64Array::from(ras));
                let dec_arr = Arc::new(Float64Array::from(decs));
                let batch_out = RecordBatch::try_new(
                    Arc::clone(&shard_schema),
                    vec![
                        Arc::new(pix_arr),
                        id_arr,
                        ra_arr,
                        dec_arr,
                    ],
                )?;
                writer.write(&batch_out)?;
                written += chunk.len() as u64;
            }
            writer.close()?;
            Ok(written)
        })
        .collect_into_vec(&mut write_results);

    let rows_written: u64 = write_results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum();

    Ok((rows_written, id_b_name_owned))
}

/// Partition catalog B (single file) into HEALPix shards in a temp directory.
/// Returns (temp_dir_guard, shard_dir_path, rows_written, original_id_b_column_name).
/// Caller must keep the guard alive.
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
    let (rows_written, id_b_name) =
        partition_file_to_shard_dir(catalog_b, &shard_dir, depth, n_shards, batch_size_b, ra_col, dec_col, id_col_b, from_radians)?;
    Ok((temp_dir, shard_dir, rows_written, id_b_name))
}

/// Partition a catalog file into HEALPix shards in the given output directory.
/// Creates output_dir if needed. Writes shard_0000.parquet, ... (same layout as
/// pre-partitioned B for cross_match). Returns (rows_written, id_column_name).
pub fn partition_catalog_impl(
    catalog_path: &Path,
    output_dir: &Path,
    depth: u8,
    n_shards: usize,
    batch_size: usize,
    ra_col: &str,
    dec_col: &str,
    _id_col: Option<&str>,
    from_radians: bool,
) -> Result<(u64, String), Box<dyn std::error::Error + Send + Sync>> {
    std::fs::create_dir_all(output_dir).map_err(|e| format!("create output dir: {}", e))?;
    partition_file_to_shard_dir(
        catalog_path,
        output_dir,
        depth,
        n_shards,
        batch_size,
        ra_col,
        dec_col,
        _id_col,
        from_radians,
    )
}

/// Cone search: find all catalog rows within radius_arcsec of (ra_deg, dec_deg).
/// Streams catalog in batches, filters by haversine distance, writes output with
/// separation_arcsec column added. Returns number of rows written.
pub fn cone_search_impl(
    catalog_path: &Path,
    ra_deg: f64,
    dec_deg: f64,
    radius_arcsec: f64,
    output_path: &Path,
    ra_col: &str,
    dec_col: &str,
    _id_col: Option<&str>,
    from_radians: bool,
    batch_size: usize,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let input = crate::parquet_mmap::ParquetInput::open(catalog_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
    let input_schema = builder.schema().clone();
    let reader = builder.with_batch_size(batch_size).build()?;

    let mut writer: Option<ArrowWriter<BufWriter<File>>> = None;
    let mut total_written: u64 = 0;

    for batch_result in reader {
        let batch = batch_result?;
        if batch.num_rows() == 0 {
            continue;
        }
        let ra_idx = column_index(&batch, ra_col);
        let dec_idx = column_index(&batch, dec_col);
        let ra_dec = ra_dec_degrees(&batch, ra_idx, dec_idx, from_radians)?;
        let ra_deg_vec = ra_dec.ra();
        let dec_deg_vec = ra_dec.dec();
        let n = batch.num_rows();
        let mut sep_vec: Vec<f64> = Vec::with_capacity(n);
        let mut mask_vec: Vec<bool> = Vec::with_capacity(n);
        for i in 0..n {
            let sep = haversine_arcsec(
                ra_deg_vec[i],
                dec_deg_vec[i],
                ra_deg,
                dec_deg,
            );
            sep_vec.push(sep);
            mask_vec.push(sep <= radius_arcsec);
        }
        let any_match = mask_vec.iter().any(|&b| b);
        if !any_match {
            continue;
        }
        let mask = BooleanArray::from(mask_vec.clone());
        let filtered = filter_record_batch(&batch, &mask)?;
        let n_filtered = filtered.num_rows();
        let mut sep_filtered: Vec<f64> = Vec::with_capacity(n_filtered);
        for (i, &m) in mask_vec.iter().enumerate() {
            if m {
                sep_filtered.push(sep_vec[i]);
            }
        }
        let sep_arr = Arc::new(Float64Array::from(sep_filtered));
        let mut new_columns = filtered.columns().to_vec();
        let mut new_fields = filtered.schema().fields().to_vec();
        new_columns.push(sep_arr);
        new_fields.push(Arc::new(Field::new(
            "separation_arcsec",
            DataType::Float64,
            false,
        )));
        let extended_schema = Arc::new(Schema::new(new_fields));
        let extended_batch =
            RecordBatch::try_new(extended_schema.clone(), new_columns)?;

        if writer.is_none() {
            std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
                .map_err(|e| format!("create output dir: {}", e))?;
            let out_file = File::create(output_path)?;
            writer = Some(ArrowWriter::try_new(
                BufWriter::new(out_file),
                extended_schema,
                Some(WriterProperties::builder().build()),
            )?);
        }
        writer.as_mut().unwrap().write(&extended_batch)?;
        total_written += n_filtered as u64;
    }

    if let Some(w) = writer {
        w.close()?;
    } else {
        let schema = input_schema.as_ref();
        let empty_schema = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .cloned()
                .chain([Arc::new(Field::new(
                    "separation_arcsec",
                    DataType::Float64,
                    false,
                ))])
                .collect::<Vec<_>>(),
        ));
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| format!("create output dir: {}", e))?;
        let out_file = File::create(output_path)?;
        let mut w = ArrowWriter::try_new(
            BufWriter::new(out_file),
            empty_schema.clone(),
            Some(WriterProperties::builder().build()),
        )?;
        w.write(&RecordBatch::new_empty(empty_schema))?;
        w.close()?;
    }

    Ok(total_written)
}

/// Batch cone search: find catalog rows within at least one of the given cones.
/// Each query is (ra_deg, dec_deg, radius_arcsec). Output includes query_index (0-based).
pub fn batch_cone_search_impl(
    catalog_path: &Path,
    queries: &[(f64, f64, f64)],
    output_path: &Path,
    ra_col: &str,
    dec_col: &str,
    _id_col: Option<&str>,
    from_radians: bool,
    batch_size: usize,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    if queries.is_empty() {
        let schema = ParquetRecordBatchReaderBuilder::try_new(crate::parquet_mmap::ParquetInput::open(catalog_path)?)?
            .schema()
            .clone();
        let empty_schema = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .cloned()
                .chain([
                    Arc::new(Field::new(
                        "separation_arcsec",
                        DataType::Float64,
                        false,
                    )),
                    Arc::new(Field::new("query_index", DataType::Int32, false)),
                ])
                .collect::<Vec<_>>(),
        ));
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| format!("create output dir: {}", e))?;
        let out_file = File::create(output_path)?;
        let mut w = ArrowWriter::try_new(
            BufWriter::new(out_file),
            empty_schema.clone(),
            Some(WriterProperties::builder().build()),
        )?;
        w.write(&RecordBatch::new_empty(empty_schema))?;
        w.close()?;
        return Ok(0);
    }

    let input = crate::parquet_mmap::ParquetInput::open(catalog_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
    let input_schema = builder.schema().clone();
    let reader = builder.with_batch_size(batch_size).build()?;

    let mut writer: Option<ArrowWriter<BufWriter<File>>> = None;
    let mut total_written: u64 = 0;

    for batch_result in reader {
        let batch = batch_result?;
        if batch.num_rows() == 0 {
            continue;
        }
        let ra_idx = column_index(&batch, ra_col);
        let dec_idx = column_index(&batch, dec_col);
        let ra_dec = ra_dec_degrees(&batch, ra_idx, dec_idx, from_radians)?;
        let ra_deg_vec = ra_dec.ra();
        let dec_deg_vec = ra_dec.dec();
        let n = batch.num_rows();

        let mut best_sep: Vec<f64> = vec![f64::INFINITY; n];
        let mut best_idx: Vec<i32> = vec![-1; n];

        for (qi, &(qra, qdec, qrad)) in queries.iter().enumerate() {
            for i in 0..n {
                let sep = haversine_arcsec(
                    ra_deg_vec[i],
                    dec_deg_vec[i],
                    qra,
                    qdec,
                );
                if sep <= qrad && sep < best_sep[i] {
                    best_sep[i] = sep;
                    best_idx[i] = qi as i32;
                }
            }
        }

        let mask_vec: Vec<bool> = best_idx.iter().map(|&x| x >= 0).collect();
        let any_match = mask_vec.iter().any(|&b| b);
        if !any_match {
            continue;
        }
        let mask = BooleanArray::from(mask_vec.clone());
        let filtered = filter_record_batch(&batch, &mask)?;
        let n_filtered = filtered.num_rows();
        let mut sep_filtered: Vec<f64> = Vec::with_capacity(n_filtered);
        let mut idx_filtered: Vec<i32> = Vec::with_capacity(n_filtered);
        for (i, &m) in mask_vec.iter().enumerate() {
            if m {
                sep_filtered.push(best_sep[i]);
                idx_filtered.push(best_idx[i]);
            }
        }
        let sep_arr = Arc::new(Float64Array::from(sep_filtered));
        let idx_arr = Arc::new(Int32Array::from(idx_filtered));
        let mut new_columns = filtered.columns().to_vec();
        let mut new_fields = filtered.schema().fields().to_vec();
        new_columns.push(sep_arr);
        new_columns.push(idx_arr);
        new_fields.push(Arc::new(Field::new(
            "separation_arcsec",
            DataType::Float64,
            false,
        )));
        new_fields.push(Arc::new(Field::new("query_index", DataType::Int32, false)));
        let extended_schema = Arc::new(Schema::new(new_fields));
        let extended_batch =
            RecordBatch::try_new(extended_schema.clone(), new_columns)?;

        if writer.is_none() {
            std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
                .map_err(|e| format!("create output dir: {}", e))?;
            let out_file = File::create(output_path)?;
            writer = Some(ArrowWriter::try_new(
                BufWriter::new(out_file),
                extended_schema,
                Some(WriterProperties::builder().build()),
            )?);
        }
        writer.as_mut().unwrap().write(&extended_batch)?;
        total_written += n_filtered as u64;
    }

    if let Some(w) = writer {
        w.close()?;
    } else {
        let schema = input_schema.as_ref();
        let empty_schema = Arc::new(Schema::new(
            schema
                .fields()
                .iter()
                .cloned()
                .chain([
                    Arc::new(Field::new(
                        "separation_arcsec",
                        DataType::Float64,
                        false,
                    )),
                    Arc::new(Field::new("query_index", DataType::Int32, false)),
                ])
                .collect::<Vec<_>>(),
        ));
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| format!("create output dir: {}", e))?;
        let out_file = File::create(output_path)?;
        let mut w = ArrowWriter::try_new(
            BufWriter::new(out_file),
            empty_schema.clone(),
            Some(WriterProperties::builder().build()),
        )?;
        w.write(&RecordBatch::new_empty(empty_schema))?;
        w.close()?;
    }

    Ok(total_written)
}

/// Attach ra_a, dec_a, ra_b, dec_b to matches by looking up coordinates from catalogs.
/// Reads matches, catalog_a, catalog_b; joins on id_a/id_b; writes extended matches.
/// id_col_a/id_col_b are the ID column names in the catalogs; matches columns are inferred
/// (first non-separation_arcsec = id_a, second = id_b).
pub fn attach_match_coords_impl(
    matches_path: &Path,
    catalog_a_path: &Path,
    catalog_b_path: &Path,
    output_path: &Path,
    id_col_a: Option<&str>,
    id_col_b: Option<&str>,
    ra_col: &str,
    dec_col: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let build_id_to_ra_dec = |path: &Path,
                              id_col_name: &str,
                              ra_col: &str,
                              dec_col: &str|
     -> Result<HashMap<IdVal, (f64, f64)>, Box<dyn std::error::Error + Send + Sync>> {
        let input = crate::parquet_mmap::ParquetInput::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(input)?.build()?;
        let mut map = HashMap::new();
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let id_idx = column_index(&batch, id_col_name);
            let ra_idx = column_index(&batch, ra_col);
            let dec_idx = column_index(&batch, dec_col);
            let ra_dec = ra_dec_degrees(&batch, ra_idx, dec_idx, false)?;
            let ra_vec = ra_dec.ra();
            let dec_vec = ra_dec.dec();
            for i in 0..batch.num_rows() {
                let id_val = get_id_value(&batch, id_col_name, id_idx, i);
                map.insert(id_val, (ra_vec[i], dec_vec[i]));
            }
        }
        Ok(map)
    };

    let infer_id = |path: &Path, ra_col: &str, dec_col: &str| -> String {
        if let Ok(input) = crate::parquet_mmap::ParquetInput::open(path) {
            if let Ok(b) = ParquetRecordBatchReaderBuilder::try_new(input) {
                let schema = b.schema();
                return id_column_from_schema(schema.as_ref(), ra_col, dec_col, None);
            }
        }
        "id".to_string()
    };
    let id_a_cat = id_col_a
        .map(str::to_string)
        .unwrap_or_else(|| infer_id(catalog_a_path, ra_col, dec_col));
    let id_b_cat = id_col_b
        .map(str::to_string)
        .unwrap_or_else(|| infer_id(catalog_b_path, ra_col, dec_col));
    let a_map = build_id_to_ra_dec(catalog_a_path, &id_a_cat, ra_col, dec_col)?;
    let b_map = build_id_to_ra_dec(catalog_b_path, &id_b_cat, ra_col, dec_col)?;

    let input_m = crate::parquet_mmap::ParquetInput::open(matches_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input_m)?.build()?;
    let schema = reader.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let id_a_name = names
        .iter()
        .find(|&&n| n != "separation_arcsec")
        .copied()
        .unwrap_or("id_a");
    let id_b_name = names
        .iter()
        .find(|&&n| n != "separation_arcsec" && n != id_a_name)
        .copied()
        .unwrap_or("id_b");

    let mut batches_out: Vec<RecordBatch> = Vec::new();

    for batch in reader {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let id_a_idx = column_index(&batch, id_a_name);
        let id_b_idx = column_index(&batch, id_b_name);
        let n = batch.num_rows();
        let mut ra_a_vals: Vec<f64> = Vec::with_capacity(n);
        let mut dec_a_vals: Vec<f64> = Vec::with_capacity(n);
        let mut ra_b_vals: Vec<f64> = Vec::with_capacity(n);
        let mut dec_b_vals: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let id_a_val = get_id_value(&batch, id_a_name, id_a_idx, i);
            let id_b_val = get_id_value(&batch, id_b_name, id_b_idx, i);
            let (r_a, d_a) = a_map.get(&id_a_val).copied().unwrap_or((f64::NAN, f64::NAN));
            let (r_b, d_b) = b_map.get(&id_b_val).copied().unwrap_or((f64::NAN, f64::NAN));
            ra_a_vals.push(r_a);
            dec_a_vals.push(d_a);
            ra_b_vals.push(r_b);
            dec_b_vals.push(d_b);
        }
        let ra_a_arr = Arc::new(Float64Array::from(ra_a_vals));
        let dec_a_arr = Arc::new(Float64Array::from(dec_a_vals));
        let ra_b_arr = Arc::new(Float64Array::from(ra_b_vals));
        let dec_b_arr = Arc::new(Float64Array::from(dec_b_vals));
        let mut new_cols = batch.columns().to_vec();
        let mut new_fields = batch.schema().fields().to_vec();
        new_cols.push(ra_a_arr);
        new_cols.push(dec_a_arr);
        new_cols.push(ra_b_arr);
        new_cols.push(dec_b_arr);
        new_fields.push(Arc::new(Field::new("ra_a", DataType::Float64, false)));
        new_fields.push(Arc::new(Field::new("dec_a", DataType::Float64, false)));
        new_fields.push(Arc::new(Field::new("ra_b", DataType::Float64, false)));
        new_fields.push(Arc::new(Field::new("dec_b", DataType::Float64, false)));
        let out_batch = RecordBatch::try_new(Arc::new(Schema::new(new_fields)), new_cols)?;
        batches_out.push(out_batch);
    }

    if batches_out.is_empty() {
        let mut empty_fields = schema.fields().to_vec();
        empty_fields.push(Arc::new(Field::new("ra_a", DataType::Float64, false)));
        empty_fields.push(Arc::new(Field::new("dec_a", DataType::Float64, false)));
        empty_fields.push(Arc::new(Field::new("ra_b", DataType::Float64, false)));
        empty_fields.push(Arc::new(Field::new("dec_b", DataType::Float64, false)));
        let empty_schema = Arc::new(Schema::new(empty_fields));
        std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| format!("create output dir: {}", e))?;
        let out_file = File::create(output_path)?;
        let mut w = ArrowWriter::try_new(
            BufWriter::new(out_file),
            empty_schema.clone(),
            Some(WriterProperties::builder().build()),
        )?;
        w.write(&RecordBatch::new_empty(empty_schema))?;
        w.close()?;
        return Ok(());
    }

    let out_schema = batches_out[0].schema();
    std::fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))
        .map_err(|e| format!("create output dir: {}", e))?;
    let out_file = File::create(output_path)?;
    let mut w = ArrowWriter::try_new(
        BufWriter::new(out_file),
        out_schema.clone(),
        Some(WriterProperties::builder().build()),
    )?;
    for b in batches_out {
        w.write(&b)?;
    }
    w.close()?;
    Ok(())
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
    let input_b = crate::parquet_mmap::ParquetInput::open(catalog_b)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_b)?;
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
        let ra_dec = ra_dec_degrees(batch, ra_idx, dec_idx, from_radians)?;
        let ra_deg = ra_dec.ra();
        let dec_deg = ra_dec.dec();
        let n = batch.num_rows();
        let mut row_results: Vec<(usize, u64, IdVal, f64, f64)> = Vec::with_capacity(n);
        (0..n)
            .into_par_iter()
            .map(|row| {
                let lon = ra_deg[row] * DEG_TO_RAD;
                let lat = dec_deg[row] * DEG_TO_RAD;
                let pixel_id = layer.hash(lon, lat);
                let shard_ix = (pixel_id % n_shards as u64) as usize;
                let id_val = get_id_value(batch, &id_b_name, id_b_idx, row);
                (shard_ix, pixel_id, id_val, ra_deg[row], dec_deg[row])
            })
            .collect_into_vec(&mut row_results);
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
    pixels_wanted: &FxHashSet<u64>,
    n_shards: usize,
) -> Vec<(IdVal, f64, f64)> {
    let shard_indices: Vec<usize> = pixels_wanted
        .iter()
        .map(|&p| (p % n_shards as u64) as usize)
        .collect::<FxHashSet<_>>()
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
) -> FxHashMap<u64, Vec<usize>> {
    let layer = get(depth);
    let mut by_pixel: FxHashMap<u64, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(estimated_pixels(ra_b.len()), Default::default());
    for (b_row_ix, (&ra, &dec)) in ra_b.iter().zip(dec_b.iter()).enumerate() {
        let pix = layer.hash(ra * DEG_TO_RAD, dec * DEG_TO_RAD);
        by_pixel.entry(pix).or_default().push(b_row_ix);
    }
    by_pixel
}

/// Block size for dec-based bounding-box coarse filter.
const DEC_BLOCK_SIZE: usize = 64;

/// Min candidates to consider R-tree (when rtree feature and strategy=rtree).
const RTREE_CANDIDATE_THRESHOLD: usize = 2000;

/// Join strategy: PLEIADES_JOIN_STRATEGY=bdynamic|adynamic|rtree (default bdynamic).
/// rtree requires building with --features rtree.
fn join_strategy() -> String {
    env::var("PLEIADES_JOIN_STRATEGY").unwrap_or_else(|_| "bdynamic".to_string())
}

/// CPU join: columnar B, group by pixel (reuse pixels_to_look per pixel), cheap reject a≤a_max,
/// optional dot-product path for small radius, dec pre-filter with binary search, block bbox filter.
/// Returns per-row matches (candidate_ix, b_row_ix, sep_arcsec).
fn run_cpu_join(
    b_cols: &BColumns,
    index_ref: &FxHashMap<u64, Vec<usize>>,
    ra_flat_ref: &[f64],
    dec_flat_ref: &[f64],
    radius_arcsec: f64,
    radius_deg: f64,
    depth_local: u8,
) -> Vec<Vec<(usize, usize, f64)>> {
    let ra_b = &b_cols.ra_b[..];
    let dec_b = &b_cols.dec_b[..];

    let t_group = std::time::Instant::now();
    let by_pixel = group_b_by_pixel(ra_b, dec_b, depth_local);
    if env::var("PLEIADES_PROFILE").is_ok() {
        profile_log("join_group_b", t_group.elapsed().as_secs_f64(), &format!("({} pixels)", by_pixel.len()));
    }

    let radius_rad = radius_arcsec / RAD_TO_ARCSEC;
    let a_max = (radius_rad * 0.5).sin().powi(2);
    let use_dot_product = radius_arcsec < 600.0; // 10 arcmin: dot product is faster for small angles
    let cos_radius_rad = radius_rad.cos();

    let t_per_pixel = std::time::Instant::now();
    // Convert to Vec for IndexedParallelIterator (enables collect_into_vec with prealloc).
    let by_pixel_vec: Vec<(u64, Vec<usize>)> = by_pixel.into_iter().collect();
    let mut per_pixel: Vec<Vec<(usize, Vec<(usize, usize, f64)>)>> =
        Vec::with_capacity(by_pixel_vec.len());
    by_pixel_vec
        .par_iter()
        .map(|(pixel, b_indices)| {
            let pixels_to_look = cached_neighbours(depth_local, *pixel);

            // Merge and dedup candidate indices (HashSet avoids sort+dedup O(n log n) -> O(n)).
            let mut candidates_set: FxHashSet<usize> = FxHashSet::default();
            for pix in &pixels_to_look {
                if let Some(ixs) = index_ref.get(pix) {
                    candidates_set.extend(ixs.iter().copied());
                }
            }

            // Pre-filter: sort candidates by dec for binary-search dec window.
            let mut candidates_by_dec: Vec<(f64, usize)> = candidates_set
                .iter()
                .map(|&ix| (dec_flat_ref[ix], ix))
                .collect();
            if candidates_by_dec.len() > 10_000 {
                candidates_by_dec
                    .par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            } else {
                candidates_by_dec
                    .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            }

            let mut out: Vec<(usize, Vec<(usize, usize, f64)>)> = Vec::with_capacity(b_indices.len());
            if candidates_by_dec.is_empty() {
                for &b_row_ix in b_indices {
                    out.push((b_row_ix, vec![]));
                }
                return out;
            }

            let strategy = join_strategy();
            let use_adynamic = strategy == "adynamic" && candidates_by_dec.len() < b_indices.len();
            #[cfg_attr(not(feature = "rtree"), allow(unused_variables))]
            let use_rtree = strategy == "rtree"
                && candidates_by_dec.len() >= RTREE_CANDIDATE_THRESHOLD;

            #[cfg(feature = "rtree")]
            if use_rtree {
                // R-tree: build spatial index on A candidates, range-query per B row.
                let points: Vec<GeomWithData<[f64; 2], usize>> = candidates_by_dec
                    .iter()
                    .map(|&(dec, ix)| {
                        let ra = ra_flat_ref[ix];
                        GeomWithData::new([ra, dec], ix)
                    })
                    .collect();
                let tree: RTree<GeomWithData<[f64; 2], usize>> = RTree::bulk_load(points);

                for &b_row_ix in b_indices {
                    let ra_b_deg = ra_b[b_row_ix];
                    let dec_b_deg = dec_b[b_row_ix];
                    let delta_ra = radius_deg / (dec_b_deg * DEG_TO_RAD).cos().max(0.01);
                    let lower = [ra_b_deg - delta_ra, dec_b_deg - radius_deg];
                    let upper = [ra_b_deg + delta_ra, dec_b_deg + radius_deg];
                    let envelope = AABB::from_corners(lower, upper);
                    let mut row_matches = Vec::new();
                    for geom in tree.locate_in_envelope_intersecting(&envelope) {
                        let candidate_ix = geom.data;
                        let ra_a = ra_flat_ref[candidate_ix];
                        let dec_a = dec_flat_ref[candidate_ix];
                        if cheap_reject_deg(ra_a, dec_a, ra_b_deg, dec_b_deg, radius_deg) {
                            continue;
                        }
                        let sep = if use_dot_product {
                            let (x1, y1, z1) = ra_dec_to_xyz_deg(ra_a, dec_a);
                            let (x2, y2, z2) = ra_dec_to_xyz_deg(ra_b_deg, dec_b_deg);
                            let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
                            if dot >= cos_radius_rad {
                                dot.acos() * RAD_TO_ARCSEC
                            } else {
                                continue;
                            }
                        } else {
                            let lon1 = ra_a * DEG_TO_RAD;
                            let lat1 = dec_a * DEG_TO_RAD;
                            let lon2 = ra_b_deg * DEG_TO_RAD;
                            let lat2 = dec_b_deg * DEG_TO_RAD;
                            let a = haversine_a_rad(lon1, lat1, lon2, lat2);
                            if a <= a_max {
                                haversine_a_to_arcsec(a)
                            } else {
                                continue;
                            }
                        };
                        row_matches.push((candidate_ix, b_row_ix, sep));
                    }
                    out.push((b_row_ix, row_matches));
                }
                return out;
            }

            if use_adynamic {
                // A-driven: iterate A candidates (outer), for each find B rows in dec window.
                let mut b_by_dec: Vec<(f64, usize)> = b_indices
                    .iter()
                    .map(|&bi| (dec_b[bi], bi))
                    .collect();
                b_by_dec.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                let mut row_matches: Vec<Vec<(usize, usize, f64)>> =
                    (0..b_cols.len()).map(|_| Vec::new()).collect();

                for &(dec_a, candidate_ix) in &candidates_by_dec {
                    let ra_a = ra_flat_ref[candidate_ix];
                    let dec_lo = dec_a - radius_deg;
                    let dec_hi = dec_a + radius_deg;
                    let first_b = b_by_dec.partition_point(|(d, _)| *d < dec_lo);
                    let last_b = b_by_dec.partition_point(|(d, _)| *d <= dec_hi).saturating_sub(1);
                    if first_b > last_b {
                        continue;
                    }
                    let (x1, y1, z1) = if use_dot_product {
                        ra_dec_to_xyz_deg(ra_a, dec_a)
                    } else {
                        (0.0, 0.0, 0.0)
                    };
                    for (_, b_row_ix) in b_by_dec[first_b..=last_b].iter() {
                        let ra_b_deg = ra_b[*b_row_ix];
                        let dec_b_deg = dec_b[*b_row_ix];
                        if cheap_reject_deg(ra_a, dec_a, ra_b_deg, dec_b_deg, radius_deg) {
                            continue;
                        }
                        let sep = if use_dot_product {
                            let (x2, y2, z2) = ra_dec_to_xyz_deg(ra_b_deg, dec_b_deg);
                            let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
                            if dot >= cos_radius_rad {
                                dot.acos() * RAD_TO_ARCSEC
                            } else {
                                continue;
                            }
                        } else {
                            let lon1 = ra_a * DEG_TO_RAD;
                            let lat1 = dec_a * DEG_TO_RAD;
                            let lon2 = ra_b_deg * DEG_TO_RAD;
                            let lat2 = dec_b_deg * DEG_TO_RAD;
                            let a = haversine_a_rad(lon1, lat1, lon2, lat2);
                            if a <= a_max {
                                haversine_a_to_arcsec(a)
                            } else {
                                continue;
                            }
                        };
                        row_matches[*b_row_ix].push((candidate_ix, *b_row_ix, sep));
                    }
                }
                for &b_row_ix in b_indices {
                    let m = std::mem::take(&mut row_matches[b_row_ix]);
                    out.push((b_row_ix, m));
                }
                return out;
            }

            // B-driven (default): Block bbox (dec_min, dec_max, start_ix, end_ix) for each block.
            let n_blocks = (candidates_by_dec.len() + DEC_BLOCK_SIZE - 1) / DEC_BLOCK_SIZE;
            let block_bounds: Vec<(f64, f64, usize, usize)> = (0..n_blocks)
                .map(|bi| {
                    let start = bi * DEC_BLOCK_SIZE;
                    let end = (start + DEC_BLOCK_SIZE).min(candidates_by_dec.len());
                    let dec_min = candidates_by_dec[start].0;
                    let dec_max = candidates_by_dec[end - 1].0;
                    (dec_min, dec_max, start, end)
                })
                .collect();

            let dec_min_global = candidates_by_dec.first().map(|x| x.0).unwrap_or(0.0);
            let dec_max_global = candidates_by_dec.last().map(|x| x.0).unwrap_or(0.0);

            for &b_row_ix in b_indices {
                let ra_b_deg = ra_b[b_row_ix];
                let dec_b_deg = dec_b[b_row_ix];
                let lon2 = ra_b_deg * DEG_TO_RAD;
                let lat2 = dec_b_deg * DEG_TO_RAD;
                let cos_lat2 = lat2.cos();

                let (x2, y2, z2) = if use_dot_product {
                    ra_dec_to_xyz_deg(ra_b_deg, dec_b_deg)
                } else {
                    (0.0, 0.0, 0.0)
                };

                // Dec window: [dec_b - radius_deg, dec_b + radius_deg]
                let dec_lo = dec_b_deg - radius_deg;
                let dec_hi = dec_b_deg + radius_deg;

                // Skip entirely if no overlap with global dec range.
                if dec_hi < dec_min_global || dec_lo > dec_max_global {
                    out.push((b_row_ix, vec![]));
                    continue;
                }

                // Find block range that overlaps dec window (block bbox coarse filter).
                let first_block = block_bounds
                    .partition_point(|(_, dec_max, _, _)| *dec_max < dec_lo)
                    .min(n_blocks);
                let last_block = block_bounds
                    .partition_point(|(dec_min, _, _, _)| *dec_min <= dec_hi)
                    .saturating_sub(1)
                    .max(first_block);

                let mut row_matches = Vec::new();

                for bi in first_block..=last_block {
                    let (_, _, start, end) = block_bounds[bi];
                    let block_dec_min = block_bounds[bi].0;
                    let block_dec_max = block_bounds[bi].1;
                    if dec_hi < block_dec_min || dec_lo > block_dec_max {
                        continue;
                    }

                    let block_candidates = &candidates_by_dec[start..end];
                    let iter_candidates = block_candidates.iter().filter(|(dec, _)| {
                        *dec >= dec_lo && *dec <= dec_hi
                    }).map(|(_, ix)| *ix);

                    let cand_vec: Vec<usize> = iter_candidates.collect();
                    let chunks8 = cand_vec.chunks_exact(8);
                    let remainder8 = chunks8.remainder();
                    let remainder4 = remainder8.chunks_exact(4);
                    let remainder = remainder4.remainder();

                    for chunk in chunks8 {
                        let mut ra1 = [0.0_f64; 8];
                        let mut dec1 = [0.0_f64; 8];
                        for (i, &ix) in chunk.iter().enumerate() {
                            ra1[i] = ra_flat_ref[ix];
                            dec1[i] = dec_flat_ref[ix];
                        }
                        if use_dot_product {
                            for (i, &candidate_ix) in chunk.iter().enumerate() {
                                let (x1, y1, z1) = ra_dec_to_xyz_deg(ra1[i], dec1[i]);
                                let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
                                if dot >= cos_radius_rad {
                                    let sep = dot.acos() * RAD_TO_ARCSEC;
                                    row_matches.push((candidate_ix, b_row_ix, sep));
                                }
                            }
                        } else {
                            let mut a_vals = [0.0_f64; 8];
                            haversine_a_8_rad(&ra1, &dec1, lon2, lat2, cos_lat2, &mut a_vals);
                            for (i, &candidate_ix) in chunk.iter().enumerate() {
                                if a_vals[i] <= a_max {
                                    let sep = haversine_a_to_arcsec(a_vals[i]);
                                    row_matches.push((candidate_ix, b_row_ix, sep));
                                }
                            }
                        }
                    }
                    for chunk in remainder4 {
                        let mut ra1 = [0.0_f64; 4];
                        let mut dec1 = [0.0_f64; 4];
                        for (i, &ix) in chunk.iter().enumerate() {
                            ra1[i] = ra_flat_ref[ix];
                            dec1[i] = dec_flat_ref[ix];
                        }
                        if use_dot_product {
                            for (i, &candidate_ix) in chunk.iter().enumerate() {
                                let (x1, y1, z1) = ra_dec_to_xyz_deg(ra1[i], dec1[i]);
                                let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
                                if dot >= cos_radius_rad {
                                    let sep = dot.acos() * RAD_TO_ARCSEC;
                                    row_matches.push((candidate_ix, b_row_ix, sep));
                                }
                            }
                        } else {
                            let mut a_vals = [0.0_f64; 4];
                            haversine_a_4_rad(&ra1, &dec1, lon2, lat2, cos_lat2, &mut a_vals);
                            for (i, &candidate_ix) in chunk.iter().enumerate() {
                                if a_vals[i] <= a_max {
                                    let sep = haversine_a_to_arcsec(a_vals[i]);
                                    row_matches.push((candidate_ix, b_row_ix, sep));
                                }
                            }
                        }
                    }
                    for &candidate_ix in remainder {
                        let ra_a = ra_flat_ref[candidate_ix];
                        let dec_a = dec_flat_ref[candidate_ix];
                        if cheap_reject_deg(ra_a, dec_a, ra_b_deg, dec_b_deg, radius_deg) {
                            continue;
                        }
                        if use_dot_product {
                            let (x1, y1, z1) = ra_dec_to_xyz_deg(ra_a, dec_a);
                            let dot = (x1 * x2 + y1 * y2 + z1 * z2).min(1.0).max(-1.0);
                            if dot >= cos_radius_rad {
                                let sep = dot.acos() * RAD_TO_ARCSEC;
                                row_matches.push((candidate_ix, b_row_ix, sep));
                            }
                        } else {
                            let lon1 = ra_a * DEG_TO_RAD;
                            let lat1 = dec_a * DEG_TO_RAD;
                            let a = haversine_a_rad(lon1, lat1, lon2, lat2);
                            if a <= a_max {
                                let sep = haversine_a_to_arcsec(a);
                                row_matches.push((candidate_ix, b_row_ix, sep));
                            }
                        }
                    }
                }
                out.push((b_row_ix, row_matches));
            }
            out
        })
        .collect_into_vec(&mut per_pixel);
    if env::var("PLEIADES_PROFILE").is_ok() {
        profile_log("join_haversine_loop", t_per_pixel.elapsed().as_secs_f64(), "(per-pixel)");
    }

    let mut row_matches: Vec<Vec<(usize, usize, f64)>> =
        (0..b_cols.len()).map(|_| Vec::new()).collect();
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
/// If match_callback is Some, matches are streamed to the callback and no output file is written.
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
    include_coords: bool,
    progress_callback: ProgressCallback,
    mut match_callback: MatchCallback,
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
    let a_channel_cap = if use_b_prefetch { 4 } else { 2 };
    if verbose && use_b_prefetch {
        verbose_log("B prefetch: overlap index||load B and join||load B (two requests in flight)");
    }

    // Prefetch thread: read catalog A batches one ahead (or two when B prefetch) so I/O overlaps with compute.
    let (tx_a, rx_a) = mpsc::sync_channel::<Result<Option<RecordBatch>, Box<dyn std::error::Error + Send + Sync>>>(a_channel_cap);
    let path_a = catalog_a.to_path_buf();
    let batch_size_a_prefetch = batch_size_a;
    let reader_handle = thread::spawn(move || {
        let input_a = match crate::parquet_mmap::ParquetInput::open(&path_a) {
            Ok(i) => i,
            Err(e) => {
                let _ = tx_a.send(Err(e.into()));
                return;
            }
        };
        let reader_a = match ParquetRecordBatchReaderBuilder::try_new(input_a)
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
    // Capacity 4: send current + next B requests so index||load B and join||load B overlap.
    let (tx_b_request, rx_b_request) = mpsc::sync_channel::<Option<FxHashSet<u64>>>(4);
    let (tx_b_response, rx_b_response) = mpsc::sync_channel::<BLoadResult>(4);
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
    let mut next_pixels_wanted: Option<FxHashSet<u64>> = None;
    // Prebuilt (index, pixels) for current chunk when we ran index build in parallel with previous join.
    let mut prebuilt_index_pixels: Option<(FxHashMap<u64, Vec<usize>>, FxHashSet<u64>)> = None;
    let mut index_prebuild_handle: Option<
        thread::JoinHandle<Result<(FxHashMap<u64, Vec<usize>>, FxHashSet<u64>), Box<dyn std::error::Error + Send + Sync>>>,
    > = None;

    // Reuse match vectors across chunks to avoid per-chunk allocation churn.
    let mut matches_id_a: Vec<IdVal> = Vec::new();
    let mut matches_id_b: Vec<IdVal> = Vec::new();
    let mut matches_sep: Vec<f64> = Vec::new();
    let mut matches_ra_a: Vec<f64> = Vec::new();
    let mut matches_dec_a: Vec<f64> = Vec::new();
    let mut matches_ra_b: Vec<f64> = Vec::new();
    let mut matches_dec_b: Vec<f64> = Vec::new();

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

        let ra_dec_a = ra_dec_degrees(&batch_a, ra_idx, dec_idx, from_radians)?;
        let ra_deg_a = ra_dec_a.ra();
        let dec_deg_a = ra_dec_a.dec();

        let t_pixels_index = std::time::Instant::now();
        // Consume previous iteration's index prebuild (ran in parallel with last chunk's join+write).
        if let Some(h) = index_prebuild_handle.take() {
            if let Ok(Ok((idx, px))) = h.join() {
                prebuilt_index_pixels = Some((idx, px));
            }
        }
        let (index, pixels_wanted, used_index_only) = if let Some((idx, px)) = prebuilt_index_pixels.take() {
            // Use index+pixels prebuilt in parallel with previous chunk's join.
            (idx, px, true)
        } else if let Some(reuse) = next_pixels_wanted.take() {
            // Chunk 1+ (no prebuild): reuse pixels from previous B-prefetch; only build index.
            let index = index_only(ra_deg_a, dec_deg_a, depth);
            (index, reuse, true)
        } else {
            // Chunk 0: single pass for both index and pixel set.
            let (idx, px) = pixels_and_index(ra_deg_a, dec_deg_a, depth);
            (idx, px, false)
        };
        let pixels_index_label = if used_index_only {
            " (index_only)"
        } else {
            " (pixels_and_index)"
        };

        // Send B load request for current chunk when first (no prefetch yet). For next chunk, the
        // index prebuild (spawned during join) will send it.
        if use_b_prefetch {
            if prefetched_b.is_none() {
                let _ = tx_b_request.send(Some(pixels_wanted.clone()));
            }
        }
        if verbose {
            verbose_log_timed(
                "  pixels+index",
                t_pixels_index.elapsed().as_secs_f64(),
                &format!("({} pixels){}", pixels_wanted.len(), pixels_index_label),
            );
        }

        // Build id_a_flat; use ra/dec slices directly (zero-copy when Float64+degrees).
        let mut id_a_flat: Vec<IdVal> = Vec::with_capacity(n_a);
        (0..n_a)
            .into_par_iter()
            .map(|row| get_id_value(&batch_a, &id_a_col, id_a_idx, row))
            .collect_into_vec(&mut id_a_flat);
        let ra_flat_ref = ra_deg_a;
        let dec_flat_ref = dec_deg_a;

        matches_id_a.clear();
        matches_id_b.clear();
        matches_sep.clear();
        matches_ra_a.clear();
        matches_dec_a.clear();
        matches_ra_b.clear();
        matches_dec_b.clear();

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
        // Pre-allocate match vectors to reduce reallocs during join.
        // Heuristic: ~0.1% of cross product, scaling up with larger radius; 1.5x headroom.
        let density = 0.001
            * (1.0 + (radius_arcsec / 60.0).min(50.0));
        let est_matches = ((n_a as f64 * n_b_loaded as f64 * density) as usize)
            .min(10_000_000)
            .max(256);
        let est_cap = est_matches + (est_matches >> 1);
        matches_id_a.reserve(est_cap);
        matches_id_b.reserve(est_cap);
        matches_sep.reserve(est_cap);
        if include_coords {
            matches_ra_a.reserve(est_cap);
            matches_dec_a.reserve(est_cap);
            matches_ra_b.reserve(est_cap);
            matches_dec_b.reserve(est_cap);
        }
        if verbose {
            verbose_log_timed("  load B", t_load.elapsed().as_secs_f64(), &format!("({} rows)", n_b_loaded));
        }
        // Spawn index prebuild for next chunk to run in parallel with join+write.
        if use_b_prefetch {
            if let Some(ref next) = next_batch {
                let tx_b = tx_b_request.clone();
                let ra_idx_n = column_index(next, ra_col);
                let dec_idx_n = column_index(next, dec_col);
                let ra_dec_n = ra_dec_degrees(next, ra_idx_n, dec_idx_n, from_radians)?;
                let ra_owned: Vec<f64> = ra_dec_n.ra().to_vec();
                let dec_owned: Vec<f64> = ra_dec_n.dec().to_vec();
                index_prebuild_handle = Some(thread::spawn(move || {
                    let (idx, px) = pixels_and_index(&ra_owned, &dec_owned, depth);
                    let _ = tx_b.send(Some(px.clone()));
                    Ok((idx, px))
                }));
            }
        }
        let t_join = std::time::Instant::now();
        let index_ref = &index;
        let radius = radius_arcsec;
        let depth_local = depth;

        #[cfg(feature = "wgpu")]
        {
            // Use GPU by default when wgpu feature is built and a GPU is available.
            // Set PLEIADES_GPU=0, cpu, or off to force CPU.
            let gpu_env = env::var("PLEIADES_GPU").as_deref().unwrap_or("");
            let use_gpu = !matches!(gpu_env.to_lowercase().as_str(), "0" | "off" | "cpu" | "false")
                && crate::gpu::gpu_available();
            if use_gpu {
                // Collect (a_ix, b_ix) candidate pairs from HEALPix index (no distance yet).
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for (b_row_ix, (&ra_b_deg, &dec_b_deg)) in b_cols.ra_b.iter().zip(b_cols.dec_b.iter()).enumerate() {
                    let center_pix = layer.hash(ra_b_deg * DEG_TO_RAD, dec_b_deg * DEG_TO_RAD);
                    let pixels_to_look = cached_neighbours(depth_local, center_pix);
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
                                let a_ix = a_ix as usize;
                                let b_ix = b_ix as usize;
                                matches_id_a.push(id_a_flat[a_ix].clone());
                                matches_id_b.push(b_cols.id_b[b_ix].clone());
                                matches_sep.push(sep as f64);
                                if include_coords {
                                    matches_ra_a.push(ra_flat_ref[a_ix]);
                                    matches_dec_a.push(dec_flat_ref[a_ix]);
                                    matches_ra_b.push(b_cols.ra_b[b_ix]);
                                    matches_dec_b.push(b_cols.dec_b[b_ix]);
                                }
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
                                    if include_coords {
                                        matches_ra_a.push(ra_flat_ref[candidate_ix]);
                                        matches_dec_a.push(dec_flat_ref[candidate_ix]);
                                        matches_ra_b.push(b_cols.ra_b[b_row_ix]);
                                        matches_dec_b.push(b_cols.dec_b[b_row_ix]);
                                    }
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
                            if include_coords {
                                matches_ra_a.push(ra_flat_ref[candidate_ix]);
                                matches_dec_a.push(dec_flat_ref[candidate_ix]);
                                matches_ra_b.push(b_cols.ra_b[b_row_ix]);
                                matches_dec_b.push(b_cols.dec_b[b_row_ix]);
                            }
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
                    if include_coords {
                        matches_ra_a.push(ra_flat_ref[candidate_ix]);
                        matches_dec_a.push(dec_flat_ref[candidate_ix]);
                        matches_ra_b.push(b_cols.ra_b[b_row_ix]);
                        matches_dec_b.push(b_cols.dec_b[b_row_ix]);
                    }
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        if verbose {
            verbose_log_timed("  join", t_join.elapsed().as_secs_f64(), &format!("({} matches)", matches_id_a.len()));
        }
        // Per-chunk n_nearest: keep only best n per id_a before write (apply_n_nearest still merges across chunks).
        if let Some(n) = n_nearest {
            if include_coords {
                let (a, b, s, ra_a, dec_a, ra_b, dec_b) = merge_to_n_nearest_with_coords(
                    std::mem::take(&mut matches_id_a),
                    std::mem::take(&mut matches_id_b),
                    std::mem::take(&mut matches_sep),
                    std::mem::take(&mut matches_ra_a),
                    std::mem::take(&mut matches_dec_a),
                    std::mem::take(&mut matches_ra_b),
                    std::mem::take(&mut matches_dec_b),
                    n,
                );
                matches_id_a = a;
                matches_id_b = b;
                matches_sep = s;
                matches_ra_a = ra_a;
                matches_dec_a = dec_a;
                matches_ra_b = ra_b;
                matches_dec_b = dec_b;
            } else {
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
        }
        if !matches_id_a.is_empty() && match_callback.is_none() && out_schema.is_none() {
            let (col_a, col_b) = build_id_columns(&matches_id_a, &matches_id_b);
            let dt_a = col_a.data_type().clone();
            let dt_b = col_b.data_type().clone();
            let id_b_col_name = id_b_name.as_deref().unwrap_or("id_b");
            let mut fields = vec![
                Field::new(id_a_col.as_str(), dt_a, false),
                Field::new(id_b_col_name, dt_b, false),
                Field::new("separation_arcsec", DataType::Float64, false),
            ];
            if include_coords {
                fields.push(Field::new("ra_a", DataType::Float64, false));
                fields.push(Field::new("dec_a", DataType::Float64, false));
                fields.push(Field::new("ra_b", DataType::Float64, false));
                fields.push(Field::new("dec_b", DataType::Float64, false));
            }
            out_schema = Some(Arc::new(Schema::new(fields)));
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

        if let Some(ref mut cb) = match_callback {
            if !matches_id_a.is_empty() {
                let (mut a, mut b, mut s) = (Vec::new(), Vec::new(), Vec::new());
                std::mem::swap(&mut matches_id_a, &mut a);
                std::mem::swap(&mut matches_id_b, &mut b);
                std::mem::swap(&mut matches_sep, &mut s);
                if !cb(a, b, s) {
                    cancelled = true;
                    break;
                }
            }
        } else if let Some(ref mut w) = writer {
            let t_write = std::time::Instant::now();
            let (col_a, col_b) = build_id_columns(&matches_id_a, &matches_id_b);
            let schema = out_schema.as_ref().unwrap();
            let col_a = coerce_id_column_to_type(col_a, schema.field(0).data_type())?;
            let col_b = coerce_id_column_to_type(col_b, schema.field(1).data_type())?;
            let sep_arr = Arc::new(Float64Array::from(std::mem::take(&mut matches_sep)));
            let id_b_col = id_b_name.as_deref().unwrap_or("id_b");
            let mut batch_fields = vec![
                Field::new(id_a_col.as_str(), schema.field(0).data_type().clone(), false),
                Field::new(id_b_col, schema.field(1).data_type().clone(), false),
                Field::new("separation_arcsec", DataType::Float64, false),
            ];
            let mut batch_cols: Vec<Arc<dyn Array>> = vec![col_a, col_b, sep_arr];
            if include_coords {
                batch_fields.push(Field::new("ra_a", DataType::Float64, false));
                batch_fields.push(Field::new("dec_a", DataType::Float64, false));
                batch_fields.push(Field::new("ra_b", DataType::Float64, false));
                batch_fields.push(Field::new("dec_b", DataType::Float64, false));
                batch_cols.push(Arc::new(Float64Array::from(std::mem::take(&mut matches_ra_a))));
                batch_cols.push(Arc::new(Float64Array::from(std::mem::take(&mut matches_dec_a))));
                batch_cols.push(Arc::new(Float64Array::from(std::mem::take(&mut matches_ra_b))));
                batch_cols.push(Arc::new(Float64Array::from(std::mem::take(&mut matches_dec_b))));
            }
            let batch = RecordBatch::try_new(Arc::new(Schema::new(batch_fields)), batch_cols)?;
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
            let input = crate::parquet_mmap::ParquetInput::open(path)?;
            let meta = ParquetRecordBatchReaderBuilder::try_new(input)?
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
        let mut empty_fields: Vec<Arc<arrow::datatypes::Field>> = vec![
            Arc::new(Field::new(id_a_col, dt_a, false)),
            Arc::new(Field::new(id_b_col, dt_b, false)),
            Arc::new(Field::new("separation_arcsec", DataType::Float64, false)),
        ];
        if include_coords {
            empty_fields.push(Arc::new(Field::new("ra_a", DataType::Float64, false)));
            empty_fields.push(Arc::new(Field::new("dec_a", DataType::Float64, false)));
            empty_fields.push(Arc::new(Field::new("ra_b", DataType::Float64, false)));
            empty_fields.push(Arc::new(Field::new("dec_b", DataType::Float64, false)));
        }
        let empty_schema = Arc::new(Schema::new(empty_fields));
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
        let input_out = crate::parquet_mmap::ParquetInput::open(output_path)?;
        let meta = ParquetRecordBatchReaderBuilder::try_new(input_out)?
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

/// Entry with coordinates for apply_n_nearest when schema has ra_a, dec_a, ra_b, dec_b.
#[derive(Clone)]
struct NearestEntryWithCoords {
    sep: f64,
    id_b: IdVal,
    ra_a: f64,
    dec_a: f64,
    ra_b: f64,
    dec_b: f64,
}

impl PartialEq for NearestEntryWithCoords {
    fn eq(&self, other: &Self) -> bool {
        self.sep.total_cmp(&other.sep) == Ordering::Equal
    }
}
impl Eq for NearestEntryWithCoords {}
impl PartialOrd for NearestEntryWithCoords {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NearestEntryWithCoords {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sep.total_cmp(&other.sep)
    }
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
/// Uses select_nth_unstable instead of full sort when list.len() > n (O(n) vs O(m log m)).
fn merge_to_n_nearest(
    id_a: Vec<IdVal>,
    id_b: Vec<IdVal>,
    sep: Vec<f64>,
    n: u32,
) -> (Vec<IdVal>, Vec<IdVal>, Vec<f64>) {
    let n = n as usize;
    let mut by_a: FxHashMap<IdVal, Vec<(f64, IdVal)>> = FxHashMap::default();
    for ((a, b), s) in id_a.into_iter().zip(id_b).zip(sep) {
        by_a.entry(a).or_default().push((s, b));
    }
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut out_sep = Vec::new();
    let cmp = |a: &(f64, IdVal), b: &(f64, IdVal)| a.0.total_cmp(&b.0);
    for (a, mut list) in by_a {
        let take = list.len().min(n);
        if list.len() > n {
            list.select_nth_unstable_by(n - 1, cmp);
            list.truncate(n);
        }
        // Skip sort when n<=1 (already ordered) or empty
        if list.len() > 1 {
            list.sort_by(cmp);
        }
        for (s, b) in list.into_iter().take(take) {
            out_a.push(a.clone());
            out_b.push(b);
            out_sep.push(s);
        }
    }
    (out_a, out_b, out_sep)
}

/// Like merge_to_n_nearest but also keeps ra_a, dec_a, ra_b, dec_b in sync.
/// Uses select_nth_unstable instead of full sort when list.len() > n.
fn merge_to_n_nearest_with_coords(
    id_a: Vec<IdVal>,
    id_b: Vec<IdVal>,
    sep: Vec<f64>,
    ra_a: Vec<f64>,
    dec_a: Vec<f64>,
    ra_b: Vec<f64>,
    dec_b: Vec<f64>,
    n: u32,
) -> (Vec<IdVal>, Vec<IdVal>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = n as usize;
    let mut by_a: FxHashMap<IdVal, Vec<(f64, IdVal, f64, f64, f64, f64)>> = FxHashMap::default();
    for (((((a, b), s), ra), dec), (rb, db)) in id_a
        .into_iter()
        .zip(id_b)
        .zip(sep)
        .zip(ra_a)
        .zip(dec_a)
        .zip(ra_b.into_iter().zip(dec_b))
    {
        by_a.entry(a).or_default().push((s, b, ra, dec, rb, db));
    }
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut out_sep = Vec::new();
    let mut out_ra_a = Vec::new();
    let mut out_dec_a = Vec::new();
    let mut out_ra_b = Vec::new();
    let mut out_dec_b = Vec::new();
    let cmp =
        |a: &(f64, IdVal, f64, f64, f64, f64), b: &(f64, IdVal, f64, f64, f64, f64)| a.0.total_cmp(&b.0);
    for (a, mut list) in by_a {
        let take = list.len().min(n);
        if list.len() > n {
            list.select_nth_unstable_by(n - 1, cmp);
            list.truncate(n);
        }
        if list.len() > 1 {
            list.sort_by(cmp);
        }
        for (s, b, ra, dec, rb, db) in list.into_iter().take(take) {
            out_a.push(a.clone());
            out_b.push(b);
            out_sep.push(s);
            out_ra_a.push(ra);
            out_dec_a.push(dec);
            out_ra_b.push(rb);
            out_dec_b.push(db);
        }
    }
    (out_a, out_b, out_sep, out_ra_a, out_dec_a, out_ra_b, out_dec_b)
}

/// Keep only n_nearest smallest-separation matches per id_a; overwrite output file.
/// Streams batches so memory is O(distinct id_a × n_nearest), not O(total rows).
/// When schema has ra_a, dec_a, ra_b, dec_b, preserves those columns.
fn apply_n_nearest(
    output_path: &Path,
    id_a_col: &str,
    id_b_col: &str,
    n_nearest: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let input = crate::parquet_mmap::ParquetInput::open(output_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(input)?
    .build()?;
    let schema = reader.schema().clone();
    let id_a_idx = schema.index_of(id_a_col)?;
    let id_b_idx = schema.index_of(id_b_col)?;
    let sep_idx = schema.index_of("separation_arcsec")?;
    let has_coords = schema.index_of("ra_a").is_ok()
        && schema.index_of("dec_a").is_ok()
        && schema.index_of("ra_b").is_ok()
        && schema.index_of("dec_b").is_ok();
    let n_nearest_usize = n_nearest as usize;

    if has_coords {
        let ra_a_idx = schema.index_of("ra_a")?;
        let dec_a_idx = schema.index_of("dec_a")?;
        let ra_b_idx = schema.index_of("ra_b")?;
        let dec_b_idx = schema.index_of("dec_b")?;
        let mut by_a: HashMap<IdVal, BinaryHeap<NearestEntryWithCoords>> = HashMap::new();
        for batch in reader {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let sep_arr = batch.column(sep_idx).as_any().downcast_ref::<Float64Array>()
                .ok_or("separation_arcsec not Float64")?;
            let ra_a_arr = batch.column(ra_a_idx).as_any().downcast_ref::<Float64Array>()
                .ok_or("ra_a not Float64")?;
            let dec_a_arr = batch.column(dec_a_idx).as_any().downcast_ref::<Float64Array>()
                .ok_or("dec_a not Float64")?;
            let ra_b_arr = batch.column(ra_b_idx).as_any().downcast_ref::<Float64Array>()
                .ok_or("ra_b not Float64")?;
            let dec_b_arr = batch.column(dec_b_idx).as_any().downcast_ref::<Float64Array>()
                .ok_or("dec_b not Float64")?;
            for i in 0..batch.num_rows() {
                let id_a = get_id_value(&batch, id_a_col, id_a_idx, i);
                let id_b = get_id_value(&batch, id_b_col, id_b_idx, i);
                let sep = sep_arr.value(i);
                let entry = NearestEntryWithCoords {
                    sep,
                    id_b,
                    ra_a: ra_a_arr.value(i),
                    dec_a: dec_a_arr.value(i),
                    ra_b: ra_b_arr.value(i),
                    dec_b: dec_b_arr.value(i),
                };
                let heap = by_a.entry(id_a).or_default();
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
        let mut out_ra_a: Vec<f64> = Vec::new();
        let mut out_dec_a: Vec<f64> = Vec::new();
        let mut out_ra_b: Vec<f64> = Vec::new();
        let mut out_dec_b: Vec<f64> = Vec::new();
        for (id_a, heap) in by_a {
            let mut list: Vec<_> = heap.into_iter().collect();
            list.sort_by(|a, b| a.sep.total_cmp(&b.sep));
            for e in list.into_iter().take(n_nearest_usize) {
                out_id_a.push(id_a.clone());
                out_id_b.push(e.id_b);
                out_sep.push(e.sep);
                out_ra_a.push(e.ra_a);
                out_dec_a.push(e.dec_a);
                out_ra_b.push(e.ra_b);
                out_dec_b.push(e.dec_b);
            }
        }
        let (col_a, col_b) = build_id_columns(&out_id_a, &out_id_b);
        let batch_cols: Vec<Arc<dyn Array>> = vec![
            col_a,
            col_b,
            Arc::new(Float64Array::from(out_sep)),
            Arc::new(Float64Array::from(out_ra_a)),
            Arc::new(Float64Array::from(out_dec_a)),
            Arc::new(Float64Array::from(out_ra_b)),
            Arc::new(Float64Array::from(out_dec_b)),
        ];
        let out_batch = RecordBatch::try_new(schema.clone(), batch_cols)?;
        let out_file = File::create(output_path)?;
        let buf = BufWriter::new(out_file);
        let mut w = ArrowWriter::try_new(buf, schema, Some(WriterProperties::builder().build()))?;
        w.write(&out_batch)?;
        w.close()?;
        return Ok(());
    }

    // No coord columns
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
            let entry = NearestEntry { sep, id_b };
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
