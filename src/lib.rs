//! AstroJoin Rust engine: out-of-core spatial cross-match for astronomical catalogs.
//!
//! Built as a Python extension via PyO3/Maturin. Uses cdshealpix (HEALPix),
//! Arrow/Parquet for streaming I/O, and haversine for angular distance.

pub mod engine;

#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Cross-match two Parquet catalogs by angular distance (Rust engine).
/// catalog_b can be a file path or a directory of pre-partitioned shards (shard_*.parquet).
/// Returns a dict compatible with CrossMatchResult (output_path, rows_a_read, rows_b_read,
/// matches_count, chunks_processed, time_seconds).
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (catalog_a, catalog_b, radius_arcsec, output_path, depth=8, batch_size_a=100_000, batch_size_b=100_000, ra_col="ra", dec_col="dec", id_col_a=None, id_col_b=None, ra_dec_units="deg", n_nearest=None, progress_callback=None))]
fn cross_match(
    py: Python<'_>,
    catalog_a: &str,
    catalog_b: &str,
    radius_arcsec: f64,
    output_path: &str,
    depth: u8,
    batch_size_a: usize,
    batch_size_b: usize,
    ra_col: &str,
    dec_col: &str,
    id_col_a: Option<&str>,
    id_col_b: Option<&str>,
    ra_dec_units: &str,
    n_nearest: Option<u32>,
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
                let _ = cb.call1(
                    py,
                    (
                        chunk_ix as i64,
                        total.map(|t| t as i64),
                        rows_a as i64,
                        matches_count as i64,
                    ),
                );
            });
        }))
    } else {
        None
    };

    let stats = engine::cross_match_impl(
        path_a,
        path_b,
        path_out,
        radius_arcsec,
        depth,
        batch_size_a,
        batch_size_b,
        ra_col,
        dec_col,
        id_col_a,
        id_col_b,
        ra_dec_units,
        n_nearest,
        progress,
    )
    .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("output_path", stats.output_path)?;
    dict.set_item("rows_a_read", stats.rows_a_read as i64)?;
    dict.set_item("rows_b_read", stats.rows_b_read as i64)?;
    dict.set_item("matches_count", stats.matches_count as i64)?;
    dict.set_item("chunks_processed", stats.chunks_processed as i64)?;
    dict.set_item("time_seconds", stats.time_seconds)?;
    Ok(dict.into_py(py))
}

/// Python module entry point.
#[cfg(feature = "python")]
#[pymodule]
fn astrojoin_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cross_match, m)?)?;
    Ok(())
}
