//! Integration tests for the Rust cross-match engine: Parquet I/O and cross_match_impl.

use std::fs::File;
use std::sync::Arc;

use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pleiades_core::engine::cross_match_impl;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

fn write_parquet(path: &std::path::Path, batch: &RecordBatch) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(
        file,
        batch.schema(),
        Some(WriterProperties::builder().build()),
    )?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Two catalogs with one pair at same position: expect exactly one match with separation ~0.
#[test]
fn test_cross_match_one_pair_same_position() {
    let tmp = TempDir::new().unwrap();
    let path_a = tmp.path().join("catalog_a.parquet");
    let path_b = tmp.path().join("catalog_b.parquet");
    let path_out = tmp.path().join("matches.parquet");

    let schema_a = Schema::new(vec![
        Field::new("source_id", DataType::Int64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_a = RecordBatch::try_new(
        Arc::new(schema_a),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(Float64Array::from(vec![10.0])),
            Arc::new(Float64Array::from(vec![-5.0])),
        ],
    )
    .unwrap();
    write_parquet(&path_a, &batch_a).unwrap();

    let schema_b = Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_b = RecordBatch::try_new(
        Arc::new(schema_b),
        vec![
            Arc::new(StringArray::from(vec!["B1"])),
            Arc::new(Float64Array::from(vec![10.0])),
            Arc::new(Float64Array::from(vec![-5.0])),
        ],
    )
    .unwrap();
    write_parquet(&path_b, &batch_b).unwrap();

    cross_match_impl(
        &path_a,
        &path_b,
        &path_out,
        10.0,
        8,
        1000,
        1000,
        512,
        "ra",
        "dec",
        None,
        None,
        "deg",
        None,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    let file_out = File::open(&path_out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file_out)
        .unwrap()
        .with_batch_size(1024)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1, "expected one output batch");
    let out = &batches[0];
    assert_eq!(out.num_rows(), 1, "expected one match");
    assert_eq!(out.num_columns(), 3, "id_a, id_b, separation_arcsec");
}

/// Two catalogs with no overlapping positions: expect empty output with schema.
#[test]
fn test_cross_match_no_matches() {
    let tmp = TempDir::new().unwrap();
    let path_a = tmp.path().join("catalog_a.parquet");
    let path_b = tmp.path().join("catalog_b.parquet");
    let path_out = tmp.path().join("matches.parquet");

    let schema_a = Schema::new(vec![
        Field::new("source_id", DataType::Int64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_a = RecordBatch::try_new(
        Arc::new(schema_a),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ],
    )
    .unwrap();
    write_parquet(&path_a, &batch_a).unwrap();

    let schema_b = Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_b = RecordBatch::try_new(
        Arc::new(schema_b),
        vec![
            Arc::new(StringArray::from(vec!["B1"])),
            Arc::new(Float64Array::from(vec![180.0])),
            Arc::new(Float64Array::from(vec![90.0])),
        ],
    )
    .unwrap();
    write_parquet(&path_b, &batch_b).unwrap();

    cross_match_impl(
        &path_a,
        &path_b,
        &path_out,
        1.0,
        8,
        1000,
        1000,
        512,
        "ra",
        "dec",
        None,
        None,
        "deg",
        None,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    let file_out = File::open(&path_out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file_out)
        .unwrap()
        .with_batch_size(1024)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    // Parquet may write 0 row groups for empty table, or 1 batch with 0 rows.
    assert!(
        batches.is_empty() || (batches.len() == 1 && batches[0].num_rows() == 0),
        "expected 0 batches or 1 empty batch, got {} batch(es)",
        batches.len()
    );
    if let Some(out) = batches.first() {
        assert_eq!(out.num_columns(), 3);
    }
}

/// Two catalogs with one pair within radius (small offset): one match, separation > 0.
#[test]
fn test_cross_match_one_pair_within_radius() {
    let tmp = TempDir::new().unwrap();
    let path_a = tmp.path().join("catalog_a.parquet");
    let path_b = tmp.path().join("catalog_b.parquet");
    let path_out = tmp.path().join("matches.parquet");

    // A: (0, 0); B: (0, 0.001 deg) ~ 3.6 arcsec
    let schema_a = Schema::new(vec![
        Field::new("source_id", DataType::Int64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_a = RecordBatch::try_new(
        Arc::new(schema_a),
        vec![
            Arc::new(Int64Array::from(vec![42_i64])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.0])),
        ],
    )
    .unwrap();
    write_parquet(&path_a, &batch_a).unwrap();

    let schema_b = Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_b = RecordBatch::try_new(
        Arc::new(schema_b),
        vec![
            Arc::new(StringArray::from(vec!["B42"])),
            Arc::new(Float64Array::from(vec![0.0])),
            Arc::new(Float64Array::from(vec![0.001])),
        ],
    )
    .unwrap();
    write_parquet(&path_b, &batch_b).unwrap();

    cross_match_impl(
        &path_a,
        &path_b,
        &path_out,
        5.0,
        8,
        1000,
        1000,
        512,
        "ra",
        "dec",
        None,
        None,
        "deg",
        None,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    let file_out = File::open(&path_out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file_out)
        .unwrap()
        .with_batch_size(1024)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1);
    let out = &batches[0];
    assert_eq!(out.num_rows(), 1);
    let sep_col = out.column(2);
    let sep = sep_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let sep_val = sep.value(0);
    assert!(sep_val > 0.0 && sep_val <= 5.0, "separation should be in (0, 5] arcsec, got {}", sep_val);
}

/// Float32 ra/dec columns: engine accepts them and produces correct match.
#[test]
fn test_cross_match_float32_ra_dec() {
    let tmp = TempDir::new().unwrap();
    let path_a = tmp.path().join("catalog_a.parquet");
    let path_b = tmp.path().join("catalog_b.parquet");
    let path_out = tmp.path().join("matches.parquet");

    let schema_a = Schema::new(vec![
        Field::new("source_id", DataType::Int64, false),
        Field::new("ra", DataType::Float32, false),
        Field::new("dec", DataType::Float32, false),
    ]);
    let batch_a = RecordBatch::try_new(
        Arc::new(schema_a),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(Float32Array::from(vec![10.0_f32])),
            Arc::new(Float32Array::from(vec![-5.0_f32])),
        ],
    )
    .unwrap();
    write_parquet(&path_a, &batch_a).unwrap();

    let schema_b = Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("ra", DataType::Float32, false),
        Field::new("dec", DataType::Float32, false),
    ]);
    let batch_b = RecordBatch::try_new(
        Arc::new(schema_b),
        vec![
            Arc::new(StringArray::from(vec!["B1"])),
            Arc::new(Float32Array::from(vec![10.0_f32])),
            Arc::new(Float32Array::from(vec![-5.0_f32])),
        ],
    )
    .unwrap();
    write_parquet(&path_b, &batch_b).unwrap();

    cross_match_impl(
        &path_a,
        &path_b,
        &path_out,
        10.0,
        8,
        1000,
        1000,
        512,
        "ra",
        "dec",
        None,
        None,
        "deg",
        None,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    let file_out = File::open(&path_out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file_out)
        .unwrap()
        .with_batch_size(1024)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1, "Float32 ra/dec should yield one match");
}

/// Radians ra/dec with ra_dec_units="rad": same position in rad, expect one match.
#[test]
fn test_cross_match_radians_units() {
    let tmp = TempDir::new().unwrap();
    let path_a = tmp.path().join("catalog_a.parquet");
    let path_b = tmp.path().join("catalog_b.parquet");
    let path_out = tmp.path().join("matches.parquet");

    // 10 deg, -5 deg in radians
    let ra_rad = 10.0_f64 * std::f64::consts::PI / 180.0;
    let dec_rad = -5.0 * std::f64::consts::PI / 180.0;

    let schema_a = Schema::new(vec![
        Field::new("source_id", DataType::Int64, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_a = RecordBatch::try_new(
        Arc::new(schema_a),
        vec![
            Arc::new(Int64Array::from(vec![1_i64])),
            Arc::new(Float64Array::from(vec![ra_rad])),
            Arc::new(Float64Array::from(vec![dec_rad])),
        ],
    )
    .unwrap();
    write_parquet(&path_a, &batch_a).unwrap();

    let schema_b = Schema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("ra", DataType::Float64, false),
        Field::new("dec", DataType::Float64, false),
    ]);
    let batch_b = RecordBatch::try_new(
        Arc::new(schema_b),
        vec![
            Arc::new(StringArray::from(vec!["B1"])),
            Arc::new(Float64Array::from(vec![ra_rad])),
            Arc::new(Float64Array::from(vec![dec_rad])),
        ],
    )
    .unwrap();
    write_parquet(&path_b, &batch_b).unwrap();

    cross_match_impl(
        &path_a,
        &path_b,
        &path_out,
        10.0,
        8,
        1000,
        1000,
        512,
        "ra",
        "dec",
        None,
        None,
        "rad",
        None,
        false,
        false,
        None,
        None,
    )
    .unwrap();

    let file_out = File::open(&path_out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file_out)
        .unwrap()
        .with_batch_size(1024)
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1, "radians units should yield one match");
}
