"""Unit tests for pleiades.validation module."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pleiades.validation import (
    CatalogValidationError,
    validate_catalog_schema,
    validate_cross_match_args,
    validate_prepartitioned_dir,
)


@pytest.mark.unit
class TestValidateCatalogSchema:
    """Tests for validate_catalog_schema()."""

    def test_valid_schema_returns_schema(self, tmp_path: Path) -> None:
        """Valid catalog with ra, dec, id returns schema."""
        path = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table(
                {"source_id": [1], "ra": [0.0], "dec": [0.0]},
            ),
            path,
        )
        schema = validate_catalog_schema(path, must_have_id=True)
        assert schema is not None
        assert "ra" in schema.names
        assert "dec" in schema.names

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when path is not a file."""
        with pytest.raises(FileNotFoundError, match="Catalog not found"):
            validate_catalog_schema(tmp_path / "nonexistent.parquet")

    def test_missing_ra_column(self, tmp_path: Path) -> None:
        """Raises CatalogValidationError when ra column missing."""
        path = tmp_path / "no_ra.parquet"
        pq.write_table(pa.table({"id": [1], "dec": [0.0]}), path)
        with pytest.raises(CatalogValidationError, match="ra"):
            validate_catalog_schema(path)

    def test_missing_dec_column(self, tmp_path: Path) -> None:
        """Raises CatalogValidationError when dec column missing."""
        path = tmp_path / "no_dec.parquet"
        pq.write_table(pa.table({"id": [1], "ra": [0.0]}), path)
        with pytest.raises(CatalogValidationError, match="dec"):
            validate_catalog_schema(path)

    def test_non_numeric_ra_raises(self, tmp_path: Path) -> None:
        """Raises when ra column is not numeric."""
        path = tmp_path / "ra_str.parquet"
        pq.write_table(
            pa.table({"id": [1], "ra": ["x"], "dec": [0.0]}),
            path,
        )
        with pytest.raises(CatalogValidationError, match="numeric"):
            validate_catalog_schema(path)

    def test_id_col_provided_checked(self, tmp_path: Path) -> None:
        """When id_col provided, it must exist."""
        path = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table({"source_id": [1], "ra": [0.0], "dec": [0.0]}),
            path,
        )
        validate_catalog_schema(path, id_col="source_id")
        with pytest.raises(CatalogValidationError, match="not found"):
            validate_catalog_schema(path, id_col="missing_col")

    def test_must_have_id_no_non_radec_raises(self, tmp_path: Path) -> None:
        """Raises when must_have_id=True and only ra/dec present."""
        path = tmp_path / "only_radec.parquet"
        pq.write_table(pa.table({"ra": [0.0], "dec": [0.0]}), path)
        with pytest.raises(CatalogValidationError, match="non-ra/dec"):
            validate_catalog_schema(path, must_have_id=True)

    def test_custom_ra_dec_column_names(self, tmp_path: Path) -> None:
        """Accepts custom ra_col/dec_col names."""
        path = tmp_path / "custom.parquet"
        pq.write_table(
            pa.table({"id": [1], "RA": [0.0], "DEC": [0.0]}),
            path,
        )
        schema = validate_catalog_schema(
            path, ra_col="RA", dec_col="DEC", must_have_id=True
        )
        assert "RA" in schema.names
        assert "DEC" in schema.names

    def test_missing_column_suggests_similar(self, tmp_path: Path) -> None:
        """When column not found, error can suggest a similar name."""
        path = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table({"source_id": [1], "ra": [0.0], "dec": [0.0]}),
            path,
        )
        with pytest.raises(CatalogValidationError, match="not found"):
            validate_catalog_schema(path, ra_col="raa")  # typo for ra
        with pytest.raises(CatalogValidationError, match="Did you mean"):
            validate_catalog_schema(path, ra_col="rra")  # close to ra


@pytest.mark.unit
class TestValidateCrossMatchArgs:
    """Tests for validate_cross_match_args()."""

    def test_positive_radius_ok(self) -> None:
        """Positive radius passes."""
        validate_cross_match_args(1.0)
        validate_cross_match_args(0.1)

    def test_zero_radius_raises(self) -> None:
        """Zero radius raises."""
        with pytest.raises(CatalogValidationError, match="positive"):
            validate_cross_match_args(0.0)

    def test_negative_radius_raises(self) -> None:
        """Negative radius raises."""
        with pytest.raises(CatalogValidationError, match="positive"):
            validate_cross_match_args(-1.0)

    def test_nan_radius_raises(self) -> None:
        """NaN radius raises with clear message."""
        with pytest.raises(CatalogValidationError, match="NaN"):
            validate_cross_match_args(float("nan"))

    def test_inf_radius_raises(self) -> None:
        """Infinite radius raises."""
        with pytest.raises(CatalogValidationError, match="finite"):
            validate_cross_match_args(float("inf"))
        with pytest.raises(CatalogValidationError, match="finite"):
            validate_cross_match_args(-float("inf"))

    def test_n_nearest_one_ok(self) -> None:
        """n_nearest=1 passes."""
        validate_cross_match_args(1.0, n_nearest=1)

    def test_n_nearest_zero_raises(self) -> None:
        """n_nearest=0 raises."""
        with pytest.raises(CatalogValidationError, match="n_nearest"):
            validate_cross_match_args(1.0, n_nearest=0)

    def test_invalid_depth_raises(self) -> None:
        """depth out of range raises."""
        with pytest.raises(CatalogValidationError, match="depth"):
            validate_cross_match_args(1.0, depth=-1)
        with pytest.raises(CatalogValidationError, match="depth"):
            validate_cross_match_args(1.0, depth=20)

    def test_invalid_n_shards_raises(self) -> None:
        """n_shards < 1 raises."""
        with pytest.raises(CatalogValidationError, match="n_shards"):
            validate_cross_match_args(1.0, n_shards=0)


@pytest.mark.unit
class TestValidatePrepartitionedDir:
    """Tests for validate_prepartitioned_dir()."""

    def test_valid_shard_dir_returns_n_shards_and_schema(self, tmp_path: Path) -> None:
        """Valid shard directory returns count and schema."""
        schema = pa.schema(
            [
                ("pixel_id", pa.uint64()),
                ("id_b", pa.int64()),
                ("ra", pa.float64()),
                ("dec", pa.float64()),
            ]
        )
        for i in range(3):
            pq.write_table(
                pa.table(
                    {
                        "pixel_id": pa.array([], type=pa.uint64()),
                        "id_b": pa.array([], type=pa.int64()),
                        "ra": pa.array([], type=pa.float64()),
                        "dec": pa.array([], type=pa.float64()),
                    }
                ),
                tmp_path / f"shard_{i:04d}.parquet",
            )
        n_shards, first_schema = validate_prepartitioned_dir(tmp_path)
        assert n_shards == 3
        assert first_schema.names == schema.names

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        """Raises when path is not a directory."""
        f = tmp_path / "file.parquet"
        f.write_text("x")
        with pytest.raises(CatalogValidationError, match="not a directory"):
            validate_prepartitioned_dir(f)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """Raises when directory has no shard_*.parquet files."""
        with pytest.raises(CatalogValidationError, match="no shard_"):
            validate_prepartitioned_dir(tmp_path)

    def test_shard_missing_required_columns_raises(self, tmp_path: Path) -> None:
        """Raises when shard is missing pixel_id, id_b, ra, or dec."""
        pq.write_table(
            pa.table({"pixel_id": [0], "ra": [0.0], "dec": [0.0]}),  # no id_b
            tmp_path / "shard_0000.parquet",
        )
        with pytest.raises(CatalogValidationError, match="columns"):
            validate_prepartitioned_dir(tmp_path)
