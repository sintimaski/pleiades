//! Parallel Parquet row-group decode: decodes multiple row groups concurrently.
//!
//! Uses Rayon to split row groups across workers; each worker opens the file,
//! reads only its assigned row groups via `with_row_groups()`, and decodes to
//! RecordBatches. Falls back to sequential read when there is 0–1 row groups.

use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use std::path::Path;

use crate::parquet_mmap::{set_readahead_for_parquet_input, ParquetInput};

/// Decode all row groups from a Parquet file in parallel.
///
/// When the file has 2+ row groups, splits them across Rayon workers; each
/// worker decodes its subset independently. Returns batches in arbitrary order
/// (downstream typically processes with `par_iter` so order does not matter).
pub fn read_parquet_parallel(
    path: &Path,
    batch_size: usize,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error + Send + Sync>> {
    let input = ParquetInput::open(path)?;
    set_readahead_for_parquet_input(&input);
    let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
    let n_row_groups = builder.metadata().num_row_groups();

    if n_row_groups <= 1 {
        let mut reader = builder.with_batch_size(batch_size).build()?;
        let mut batches = Vec::new();
        for batch in reader.by_ref() {
            let b = batch?;
            if b.num_rows() > 0 {
                batches.push(b);
            }
        }
        return Ok(batches);
    }

    let path_buf = path.to_path_buf();
    let num_workers = rayon::current_num_threads().max(1);
    let chunk_size = ((n_row_groups + num_workers - 1) / num_workers).max(1);
    let indices: Vec<usize> = (0..n_row_groups).collect();
    type Err = Box<dyn std::error::Error + Send + Sync>;
    let batches: Vec<Vec<RecordBatch>> = indices
        .par_chunks(chunk_size)
        .map(|indices| -> Result<Vec<RecordBatch>, Err> {
            let mut out = Vec::new();
            let input = ParquetInput::open(&path_buf)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(input)
                .and_then(|b| b.with_row_groups(indices.to_vec()).with_batch_size(batch_size).build())?;
            for batch in reader {
                let b = batch?;
                if b.num_rows() > 0 {
                    out.push(b);
                }
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(batches.into_iter().flat_map(|v| v).collect())
}
