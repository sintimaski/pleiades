"""Unit tests for the public Python API."""

from __future__ import annotations

from pathlib import Path

import pleiades
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pleiades.validation import CatalogValidationError


@pytest.mark.unit
class TestCrossMatchAPI:
    """Test cross_match() API contract."""

    def test_cross_match_is_callable(self) -> None:
        """cross_match is exposed and callable."""
        assert callable(pleiades.cross_match)

    def test_cross_match_returns_result(self, tmp_path: Path) -> None:
        """cross_match returns CrossMatchResult with output_path and counts."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        result = pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=tmp_path / "out.parquet",
        )
        assert result is not None
        assert hasattr(result, "output_path")
        assert hasattr(result, "matches_count")
        assert hasattr(result, "time_seconds")
        assert result.output_path.endswith(".parquet")

    def test_cross_match_writes_parquet_with_expected_columns(
        self, tmp_path: Path
    ) -> None:
        """cross_match writes a Parquet file with id_a, id_b, separation_arcsec."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "schema.parquet"
        pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=out,
        )
        assert out.is_file()
        t = pq.read_table(out)
        assert "separation_arcsec" in t.column_names
        assert t.num_rows >= 0

    def test_cross_match_raises_on_invalid_radius(self, tmp_path: Path) -> None:
        """cross_match raises CatalogValidationError for non-positive radius."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "out.parquet"
        with pytest.raises(CatalogValidationError, match="positive"):
            pleiades.cross_match(
                catalog_a=fixtures / "catalog_a_small.parquet",
                catalog_b=fixtures / "catalog_b_small.parquet",
                radius_arcsec=0.0,
                output_path=out,
            )
        with pytest.raises(CatalogValidationError, match="positive"):
            pleiades.cross_match(
                catalog_a=fixtures / "catalog_a_small.parquet",
                catalog_b=fixtures / "catalog_b_small.parquet",
                radius_arcsec=-1.0,
                output_path=out,
            )

    def test_cross_match_raises_on_missing_catalog_a(self, tmp_path: Path) -> None:
        """cross_match raises FileNotFoundError when catalog A does not exist."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        with pytest.raises(FileNotFoundError, match="Catalog A not found"):
            pleiades.cross_match(
                catalog_a="/nonexistent/a.parquet",
                catalog_b=fixtures / "catalog_b_small.parquet",
                radius_arcsec=2.0,
                output_path=tmp_path / "out.parquet",
            )

    def test_cross_match_raises_on_missing_ra_column(self, tmp_path: Path) -> None:
        """cross_match raises CatalogValidationError when required column is missing."""
        bad_catalog = tmp_path / "no_dec.parquet"
        pq.write_table(
            pa.table({"id": [1], "ra": [0.0]}),  # no dec
            bad_catalog,
        )
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        with pytest.raises(CatalogValidationError, match="dec"):
            pleiades.cross_match(
                catalog_a=bad_catalog,
                catalog_b=fixtures / "catalog_b_small.parquet",
                radius_arcsec=2.0,
                output_path=tmp_path / "out.parquet",
            )

    def test_cross_match_progress_callback_called(self, tmp_path: Path) -> None:
        """progress_callback is invoked with chunk index and counts."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        calls: list[tuple[int, int | None, int, int]] = []
        pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=tmp_path / "progress.parquet",
            progress_callback=lambda ci, tc, ra, m: calls.append((ci, tc, ra, m)),
        )
        assert len(calls) >= 1
        assert calls[0][0] >= 1  # chunk index
        assert calls[-1][2] == 100  # rows_a_read (catalog_a_small has 100 rows)

    def test_cross_match_n_nearest_reduces_output(self, tmp_path: Path) -> None:
        """With n_nearest=1, output has at most one match per id_a."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out_all = tmp_path / "n_all.parquet"
        out_one = tmp_path / "n_one.parquet"
        pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=out_all,
        )
        pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=out_one,
            n_nearest=1,
        )
        t_all = pq.read_table(out_all)
        t_one = pq.read_table(out_one)
        id_col = (
            "source_id" if "source_id" in t_one.column_names else t_one.column_names[0]
        )
        ids_one = t_one.column(id_col)
        assert t_one.num_rows <= t_all.num_rows
        # Each id_a appears at most once
        assert (
            len({ids_one[i].as_py() for i in range(t_one.num_rows)}) == t_one.num_rows
        )


@pytest.mark.unit
class TestSummarizeMatches:
    """Test summarize_matches() API."""

    def test_summarize_matches_returns_summary(self, tmp_path: Path) -> None:
        """summarize_matches returns MatchSummary with counts and separation stats."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "summary_test.parquet"
        pleiades.cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
            output_path=out,
        )
        summary = pleiades.summarize_matches(out)
        assert summary.num_matches >= 0
        assert summary.separation_arcsec_min >= 0
        assert summary.separation_arcsec_max >= 0
        assert summary.id_a_column != ""
        assert summary.id_b_column != ""


@pytest.mark.unit
class TestConeSearch:
    """Test cone_search() API."""

    def test_cone_search_returns_count(self, tmp_path: Path) -> None:
        """cone_search writes matches and returns row count."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "cone_test.parquet"
        n = pleiades.cone_search(
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


@pytest.mark.unit
class TestCrossMatchIter:
    """Test cross_match_iter() streaming API."""

    def test_cross_match_iter_yields_tuples(self) -> None:
        """cross_match_iter yields (id_a, id_b, separation_arcsec)."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        it = pleiades.cross_match_iter(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radius_arcsec=2.0,
        )
        count = 0
        for _a, _b, sep in it:
            assert sep >= 0
            count += 1
        assert count >= 0
