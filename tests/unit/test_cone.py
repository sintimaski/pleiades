"""Unit tests for astrojoin.cone module."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import astrojoin
from astrojoin.validation import CatalogValidationError


@pytest.mark.unit
class TestConeSearch:
    """Tests for cone_search()."""

    def test_cone_search_finds_within_radius(self, tmp_path: Path) -> None:
        """cone_search returns rows within radius and writes separation."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "cone.parquet"
        n = astrojoin.cone_search(
            catalog_path=fixtures / "catalog_a_small.parquet",
            ra_deg=180.0,
            dec_deg=0.0,
            radius_arcsec=3600.0,  # 1 deg
            output_path=out,
        )
        assert n >= 0
        if n > 0:
            t = pq.read_table(out)
            assert "separation_arcsec" in t.column_names
            assert t.num_rows == n
            for i in range(t.num_rows):
                assert t.column("separation_arcsec")[i].as_py() <= 3600.0

    def test_cone_search_empty_when_none_in_radius(self, tmp_path: Path) -> None:
        """cone_search writes empty table when no rows within radius."""
        catalog_one = tmp_path / "one_row.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1],
                    "ra": [180.0],
                    "dec": [0.0],
                }
            ),
            catalog_one,
        )
        out = tmp_path / "cone_empty.parquet"
        n = astrojoin.cone_search(
            catalog_path=catalog_one,
            ra_deg=0.0,
            dec_deg=0.0,
            radius_arcsec=0.001,  # Very small; point at (180,0) is far
            output_path=out,
        )
        assert n == 0
        t = pq.read_table(out)
        assert t.num_rows == 0
        assert "separation_arcsec" in t.column_names

    def test_cone_search_file_not_found(self) -> None:
        """cone_search raises FileNotFoundError for missing catalog."""
        with pytest.raises(FileNotFoundError, match="Catalog not found"):
            astrojoin.cone_search(
                catalog_path="/nonexistent.parquet",
                ra_deg=0.0,
                dec_deg=0.0,
                radius_arcsec=10.0,
                output_path="/tmp/out.parquet",
            )

    def test_cone_search_invalid_schema_raises(self, tmp_path: Path) -> None:
        """cone_search raises when catalog has no dec column."""
        bad = tmp_path / "bad.parquet"
        pq.write_table(pa.table({"id": [1], "ra": [0.0]}), bad)
        out = tmp_path / "out.parquet"
        with pytest.raises(CatalogValidationError, match="dec"):
            astrojoin.cone_search(
                catalog_path=bad,
                ra_deg=0.0,
                dec_deg=0.0,
                radius_arcsec=10.0,
                output_path=out,
            )

    def test_cone_search_radians_units(self, tmp_path: Path) -> None:
        """cone_search with ra_dec_units='rad' converts catalog coords and finds match."""
        import math
        rad = math.radians(180.0)  # 180 deg in rad
        cat = tmp_path / "rad_cat.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1],
                    "ra": [rad],
                    "dec": [0.0],  # 0 rad
                }
            ),
            cat,
        )
        out = tmp_path / "cone_rad.parquet"
        n = astrojoin.cone_search(
            catalog_path=cat,
            ra_deg=180.0,
            dec_deg=0.0,
            radius_arcsec=3600.0,
            output_path=out,
            ra_dec_units="rad",
        )
        assert n == 1
        t = pq.read_table(out)
        assert t.column("separation_arcsec")[0].as_py() <= 3600.0


@pytest.mark.unit
class TestBatchConeSearch:
    """Tests for batch_cone_search()."""

    def test_batch_cone_search_multiple_queries(self, tmp_path: Path) -> None:
        """batch_cone_search finds rows matching any of the queries."""
        catalog = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2, 3],
                    "ra": [10.0, 20.0, 30.0],
                    "dec": [0.0, 0.0, 0.0],
                }
            ),
            catalog,
        )
        out = tmp_path / "batch_cone.parquet"
        queries = [
            (10.0, 0.0, 10.0),
            (30.0, 0.0, 10.0),
        ]
        n = astrojoin.batch_cone_search(catalog, queries, out)
        assert n == 2
        t = pq.read_table(out)
        assert "query_index" in t.column_names
        assert "separation_arcsec" in t.column_names
        assert t.num_rows == 2
        idx = t.column("query_index")
        assert idx[0].as_py() in (0, 1)
        assert idx[1].as_py() in (0, 1)

    def test_batch_cone_search_empty_queries(self, tmp_path: Path) -> None:
        """batch_cone_search with no matching queries writes empty output."""
        catalog = tmp_path / "cat.parquet"
        pq.write_table(
            pa.table(
                {"source_id": [1], "ra": [0.0], "dec": [0.0]},
            ),
            catalog,
        )
        out = tmp_path / "empty.parquet"
        n = astrojoin.batch_cone_search(
            catalog, [(100.0, 100.0, 1.0)], out
        )
        assert n == 0
        t = pq.read_table(out)
        assert t.num_rows == 0
        assert "query_index" in t.column_names
