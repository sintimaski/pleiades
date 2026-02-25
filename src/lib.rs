//! Pleiades Rust engine: out-of-core spatial cross-match for astronomical catalogs.
//!
//! Built as a Python extension via PyO3/Maturin. Uses cdshealpix (HEALPix),
//! Arrow/Parquet for streaming I/O, and haversine for angular distance.

// Forbid unsafe except when macos_readahead is enabled (single fcntl FFI call in engine).
#![cfg_attr(not(feature = "macos_readahead"), forbid(unsafe_code))]

pub mod engine;

#[cfg(feature = "wgpu")]
pub mod gpu;

#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
use std::sync::mpsc;

#[cfg(feature = "python")]
use std::thread;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Cross-match two Parquet catalogs by angular distance (Rust engine).
/// catalog_b can be a file path or a directory of pre-partitioned shards (shard_*.parquet).
/// Returns a dict compatible with CrossMatchResult (output_path, rows_a_read, rows_b_read,
/// matches_count, chunks_processed, time_seconds).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, radius_arcsec, output_path, depth=8, batch_size_a=100_000, batch_size_b=100_000, n_shards=512, ra_col="ra", dec_col="dec", id_col_a=None, id_col_b=None, ra_dec_units="deg", n_nearest=None, keep_b_in_memory=false, progress_callback=None))]
fn cross_match(
    py: Python<'_>,
    catalog_a: &str,
    catalog_b: &str,
    radius_arcsec: f64,
    output_path: &str,
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
    progress_callback: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let path_a = Path::new(catalog_a);
    let path_b = Path::new(catalog_b);
    let path_out = Path::new(output_path);
    if let Some(parent) = path_out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!("create output dir: {}", e))
        })?;
    }

    let progress: engine::ProgressCallback = if let Some(cb) = progress_callback {
        let cb = cb.clone().unbind();
        Some(Box::new(move |chunk_ix, total, rows_a, matches_count| {
            Python::with_gil(|py| {
                let res = cb.call1(
                    py,
                    (
                        chunk_ix as i64,
                        total.map(|t| t as i64),
                        rows_a as i64,
                        matches_count as i64,
                    ),
                );
                match res.and_then(|r| r.extract::<bool>(py)) {
                    Ok(false) => false, // cancel
                    _ => true,          // continue (None, True, or error)
                }
            })
        }))
    } else {
        None
    };

    let stats = match engine::cross_match_impl(
        path_a,
        path_b,
        path_out,
        radius_arcsec,
        depth,
        batch_size_a,
        batch_size_b,
        n_shards,
        ra_col,
        dec_col,
        id_col_a,
        id_col_b,
        ra_dec_units,
        n_nearest,
        keep_b_in_memory,
        progress,
        None,
    ) {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            if let Ok(io_err) = e.downcast::<std::io::Error>() {
                if io_err.kind() == std::io::ErrorKind::NotFound {
                    return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                        io_err.to_string(),
                    ));
                }
            }
            return Err(pyo3::exceptions::PyOSError::new_err(msg));
        }
    };

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("output_path", stats.output_path)?;
    dict.set_item("rows_a_read", stats.rows_a_read as i64)?;
    dict.set_item("rows_b_read", stats.rows_b_read as i64)?;
    dict.set_item("matches_count", stats.matches_count as i64)?;
    dict.set_item("chunks_processed", stats.chunks_processed as i64)?;
    dict.set_item("time_seconds", stats.time_seconds)?;
    Ok(dict
        .into_pyobject(py)
        .expect("PyDict into_pyobject")
        .into_any()
        .unbind())
}

/// Returns true if the extension was built with the `wgpu` feature (GPU join used by default when available; set PLEIADES_GPU=0 to force CPU).
#[cfg(feature = "python")]
#[pyfunction]
fn has_wgpu_feature() -> bool {
    cfg!(feature = "wgpu")
}

/// Batch type sent over the channel: (id_a, id_b, separation_arcsec).
type MatchBatch = (Vec<engine::IdVal>, Vec<engine::IdVal>, Vec<f64>);

/// Rust-backed streaming iterator for cross-match. Yields (id_a, id_b, separation_arcsec) as produced.
#[cfg(feature = "python")]
#[pyclass]
struct CrossMatchIter {
    receiver: std::sync::Mutex<mpsc::Receiver<Option<MatchBatch>>>,
    current: std::sync::RwLock<Option<MatchBatch>>,
    index: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "python")]
#[pymethods]
impl CrossMatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<(PyObject, PyObject, f64)> {
        loop {
            let i = slf.index.load(std::sync::atomic::Ordering::Relaxed);
            let has_more = slf.current.read().unwrap().as_ref().map_or(false, |(a, _b, _s)| i < a.len());
            if has_more {
                let (id_a, id_b, sep) = {
                    let cur = slf.current.read().unwrap();
                    let (a, b, s) = cur.as_ref().unwrap();
                    let id_a: PyObject = match &a[i] {
                        engine::IdVal::I64(x) => x.into_py(py),
                        engine::IdVal::Str(st) => st.clone().into_py(py),
                    };
                    let id_b: PyObject = match &b[i] {
                        engine::IdVal::I64(x) => x.into_py(py),
                        engine::IdVal::Str(st) => st.clone().into_py(py),
                    };
                    (id_a, id_b, s[i])
                };
                slf.index.store(i + 1, std::sync::atomic::Ordering::Relaxed);
                return Ok((id_a, id_b, sep));
            }
            match slf.receiver.lock().map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err("iterator lock poisoned")
            })?.recv() {
                Ok(Some(b)) => {
                    *slf.current.write().unwrap() = Some(b);
                    slf.index.store(0, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(None) | Err(_) => {
                    return Err(pyo3::exceptions::PyStopIteration::new_err(
                        "cross-match iteration complete",
                    ));
                }
            }
        }
    }
}

/// Stream cross-match results as (id_a, id_b, separation_arcsec). Uses Rust engine; yields as each chunk completes.
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, radius_arcsec, depth=8, batch_size_a=100_000, batch_size_b=100_000, n_shards=512, ra_col="ra", dec_col="dec", id_col_a=None, id_col_b=None, ra_dec_units="deg", n_nearest=None, keep_b_in_memory=false))]
fn cross_match_iter(
    catalog_a: &str,
    catalog_b: &str,
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
) -> PyResult<Py<PyAny>> {
    let path_a = Path::new(catalog_a).to_path_buf();
    let path_b = Path::new(catalog_b).to_path_buf();
    let (tx, rx) = mpsc::sync_channel::<Option<MatchBatch>>(4);
    let ra_col = ra_col.to_string();
    let dec_col = dec_col.to_string();
    let id_col_a = id_col_a.map(str::to_string);
    let id_col_b = id_col_b.map(str::to_string);
    let ra_dec_units = ra_dec_units.to_string();

    thread::spawn(move || {
        let out_dummy = Path::new(""); // Not used when match_callback is set
        let callback: engine::MatchCallback = Some(Box::new(move |a, b, s| {
            tx.send(Some((a, b, s))).is_ok()
        }));
        let _ = engine::cross_match_impl(
            &path_a,
            &path_b,
            out_dummy,
            radius_arcsec,
            depth,
            batch_size_a,
            batch_size_b,
            n_shards,
            &ra_col,
            &dec_col,
            id_col_a.as_deref(),
            id_col_b.as_deref(),
            &ra_dec_units,
            n_nearest,
            keep_b_in_memory,
            None,
            callback,
        );
        // Dropping tx signals completion; receiver gets RecvError
    });

    Python::with_gil(|py| {
        let iter = CrossMatchIter {
            receiver: std::sync::Mutex::new(rx),
            current: std::sync::RwLock::new(None),
            index: std::sync::atomic::AtomicUsize::new(0),
        };
        Ok(iter.into_py(py).into_any())
    })
}

/// Partition a catalog by HEALPix pixel into shard Parquet files.
/// Writes shard_0000.parquet, ... to output_dir. Returns (rows_written, id_column_name).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (catalog_path, output_dir, depth=8, n_shards=16, batch_size=250_000, ra_col="ra", dec_col="dec", id_col=None, ra_dec_units="deg"))]
fn partition_catalog(
    catalog_path: &str,
    output_dir: &str,
    depth: u8,
    n_shards: usize,
    batch_size: usize,
    ra_col: &str,
    dec_col: &str,
    id_col: Option<&str>,
    ra_dec_units: &str,
) -> PyResult<(i64, String)> {
    let path = Path::new(catalog_path);
    let out = Path::new(output_dir);
    let from_radians = ra_dec_units.to_lowercase() == "rad";
    match engine::partition_catalog_impl(path, out, depth, n_shards, batch_size, ra_col, dec_col, id_col, from_radians) {
        Ok((rows, id_name)) => Ok((rows as i64, id_name)),
        Err(e) => {
            let msg = e.to_string();
            if let Ok(io_err) = e.downcast::<std::io::Error>() {
                if io_err.kind() == std::io::ErrorKind::NotFound {
                    return Err(pyo3::exceptions::PyFileNotFoundError::new_err(msg));
                }
            }
            Err(pyo3::exceptions::PyOSError::new_err(msg))
        }
    }
}

/// Python module entry point.
#[cfg(feature = "python")]
#[pymodule]
fn pleiades_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_match, m)?)?;
    m.add_function(wrap_pyfunction!(cross_match_iter, m)?)?;
    m.add_class::<CrossMatchIter>()?;
    m.add_function(wrap_pyfunction!(partition_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(has_wgpu_feature, m)?)?;
    Ok(())
}
