"""Unit tests for pleiades.analysis module."""

from __future__ import annotations

from pathlib import Path

import pleiades
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.unit
class TestSummarizeMatches:
    """Tests for summarize_matches()."""

    def test_summarize_matches_empty_file(self, tmp_path: Path) -> None:
        """Empty matches file returns zero counts and median None."""
        empty = tmp_path / "empty.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": pa.array([], type=pa.int64()),
                    "object_id": pa.array([], type=pa.string()),
                    "separation_arcsec": pa.array([], type=pa.float64()),
                }
            ),
            empty,
        )
        summary = pleiades.summarize_matches(empty)
        assert summary.num_matches == 0
        assert summary.num_unique_id_a == 0
        assert summary.num_unique_id_b == 0
        assert summary.separation_arcsec_median is None
        assert summary.id_a_column in ("source_id", "object_id")
        assert summary.id_b_column != summary.id_a_column

    def test_summarize_matches_with_data(self, tmp_path: Path) -> None:
        """Summarize non-empty matches returns correct stats."""
        matches = tmp_path / "matches.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 1, 2],
                    "object_id": ["B1", "B2", "B3"],
                    "separation_arcsec": [0.5, 1.0, 2.0],
                }
            ),
            matches,
        )
        summary = pleiades.summarize_matches(matches)
        assert summary.num_matches == 3
        assert summary.num_unique_id_a == 2
        assert summary.num_unique_id_b == 3
        assert summary.separation_arcsec_min == 0.5
        assert summary.separation_arcsec_max == 2.0
        assert summary.separation_arcsec_mean == pytest.approx(1.166666, rel=1e-4)
        assert summary.separation_arcsec_median == 1.0
        assert summary.id_a_column == "source_id"
        assert summary.id_b_column == "object_id"

    def test_summarize_matches_file_not_found(self) -> None:
        """summarize_matches raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Matches file not found"):
            pleiades.summarize_matches("/nonexistent/matches.parquet")


@pytest.mark.unit
class TestMatchStats:
    """Tests for match_stats()."""

    def test_match_stats_returns_dict(self, tmp_path: Path) -> None:
        """match_stats returns dict with expected keys."""
        path = tmp_path / "m.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2],
                    "object_id": ["B1", "B2"],
                    "separation_arcsec": [0.5, 1.5],
                }
            ),
            path,
        )
        stats = pleiades.match_stats(path)
        assert stats["num_matches"] == 2
        assert stats["num_unique_id_a"] == 2
        assert stats["separation_arcsec_min"] == 0.5
        assert stats["separation_arcsec_max"] == 1.5
        assert "id_a_column" in stats
        assert "id_b_column" in stats

    def test_match_stats_with_percentiles(self, tmp_path: Path) -> None:
        """match_stats with separation_percentiles returns percentiles."""
        path = tmp_path / "m.parquet"
        pq.write_table(
            pa.table(
                {
                    "id_a": [1, 2, 3],
                    "id_b": ["a", "b", "c"],
                    "separation_arcsec": [1.0, 2.0, 3.0],
                }
            ),
            path,
        )
        stats = pleiades.match_stats(path, separation_percentiles=[25.0, 50.0, 75.0])
        assert "separation_percentiles" in stats
        pct = stats["separation_percentiles"]
        assert "25.0" in pct
        assert "50.0" in pct
        assert pct["50.0"] == 2.0


@pytest.mark.unit
class TestMatchQualitySummary:
    """Tests for match_quality_summary()."""

    def test_match_quality_summary_fractions(self, tmp_path: Path) -> None:
        """match_quality_summary returns fraction of A and B matched."""
        path = tmp_path / "m.parquet"
        pq.write_table(
            pa.table(
                {
                    "id_a": [1, 2],
                    "id_b": ["x", "y"],
                    "separation_arcsec": [0.5, 1.0],
                }
            ),
            path,
        )
        q = pleiades.match_quality_summary(path, rows_a=10, rows_b=5)
        assert q["num_matches"] == 2
        assert q["fraction_id_a_matched"] == 0.2
        assert q["fraction_id_b_matched"] == 0.4

    def test_match_quality_summary_empty_file(self, tmp_path: Path) -> None:
        """Empty matches returns zero fractions."""
        path = tmp_path / "empty.parquet"
        pq.write_table(
            pa.table(
                {
                    "id_a": pa.array([], type=pa.int64()),
                    "id_b": pa.array([], type=pa.string()),
                    "separation_arcsec": pa.array([], type=pa.float64()),
                }
            ),
            path,
        )
        q = pleiades.match_quality_summary(path, rows_a=100, rows_b=50)
        assert q["fraction_id_a_matched"] == 0.0
        assert q["fraction_id_b_matched"] == 0.0
        assert q["num_matches"] == 0


@pytest.mark.unit
class TestAttachMatchCoords:
    """Tests for attach_match_coords()."""

    def test_attach_match_coords_adds_columns(self, tmp_path: Path) -> None:
        """attach_match_coords adds ra_a, dec_a, ra_b, dec_b to matches."""
        matches = tmp_path / "matches.parquet"
        cat_a = tmp_path / "catalog_a.parquet"
        cat_b = tmp_path / "catalog_b.parquet"
        out = tmp_path / "out.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2],
                    "object_id": ["B1", "B2"],
                    "separation_arcsec": [0.5, 1.0],
                }
            ),
            matches,
        )
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2],
                    "ra": [10.0, 20.0],
                    "dec": [-5.0, 5.0],
                }
            ),
            cat_a,
        )
        pq.write_table(
            pa.table(
                {
                    "object_id": ["B1", "B2"],
                    "ra": [10.01, 20.01],
                    "dec": [-4.99, 5.01],
                }
            ),
            cat_b,
        )
        pleiades.attach_match_coords(
            matches,
            cat_a,
            cat_b,
            out,
            id_col_a="source_id",
            id_col_b="object_id",
        )
        t = pq.read_table(out)
        assert "ra_a" in t.column_names
        assert "dec_a" in t.column_names
        assert "ra_b" in t.column_names
        assert "dec_b" in t.column_names
        assert t.column("ra_a")[0].as_py() == 10.0
        assert t.column("ra_b")[0].as_py() == 10.01


@pytest.mark.unit
class TestMergeMatchToCatalog:
    """Tests for merge_match_to_catalog()."""

    def test_merge_match_to_catalog_left_side_a(self, tmp_path: Path) -> None:
        """Merge matches onto catalog (catalog_side=a) adds match columns."""
        matches = tmp_path / "m.parquet"
        catalog = tmp_path / "cat.parquet"
        out = tmp_path / "merged.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2],
                    "object_id": ["B1", "B2"],
                    "separation_arcsec": [0.5, 1.0],
                }
            ),
            matches,
        )
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2, 3],
                    "ra": [10.0, 20.0, 30.0],
                    "dec": [-5.0, 0.0, 5.0],
                }
            ),
            catalog,
        )
        pleiades.merge_match_to_catalog(
            matches, catalog, out, catalog_side="a", id_col_catalog="source_id"
        )
        t = pq.read_table(out)
        assert t.num_rows == 3
        assert "match_object_id" in t.column_names
        assert "separation_arcsec" in t.column_names
        # Row 0,1 have matches; row 2 (source_id=3) has null
        assert t.column("match_object_id")[0].as_py() == "B1"
        assert t.column("match_object_id")[1].as_py() == "B2"
        assert t.column("match_object_id")[2].as_py() is None

    def test_merge_match_to_catalog_side_b(self, tmp_path: Path) -> None:
        """Merge with catalog_side=b uses id_b as catalog key."""
        matches = tmp_path / "m.parquet"
        catalog = tmp_path / "cat.parquet"
        out = tmp_path / "merged.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2],
                    "object_id": ["B1", "B2"],
                    "separation_arcsec": [0.5, 1.0],
                }
            ),
            matches,
        )
        pq.write_table(
            pa.table(
                {"object_id": ["B1", "B2"], "ra": [10.0, 20.0], "dec": [0.0, 0.0]}
            ),
            catalog,
        )
        pleiades.merge_match_to_catalog(
            matches, catalog, out, catalog_side="b", id_col_catalog="object_id"
        )
        t = pq.read_table(out)
        assert t.num_rows == 2
        assert "match_source_id" in t.column_names
        assert t.column("match_source_id")[0].as_py() == 1
        assert t.column("match_source_id")[1].as_py() == 2

    def test_merge_match_to_catalog_missing_id_column(self, tmp_path: Path) -> None:
        """Raises ValueError when id_col_catalog not in catalog."""
        matches = tmp_path / "m.parquet"
        catalog = tmp_path / "cat.parquet"
        out = tmp_path / "out.parquet"
        pq.write_table(
            pa.table(
                {"source_id": [1], "object_id": ["B1"], "separation_arcsec": [0.5]}
            ),
            matches,
        )
        pq.write_table(
            pa.table({"ra": [10.0], "dec": [0.0]}),
            catalog,
        )
        with pytest.raises(ValueError, match="not in catalog columns"):
            pleiades.merge_match_to_catalog(
                matches, catalog, out, id_col_catalog="nonexistent"
            )

    def test_merge_match_to_catalog_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when matches or catalog missing."""
        cat = tmp_path / "catalog.parquet"
        matches = tmp_path / "m.parquet"
        pq.write_table(
            pa.table({"id": [1], "ra": [0.0], "dec": [0.0]}),
            cat,
        )
        pq.write_table(
            pa.table(
                {"source_id": [1], "object_id": ["B1"], "separation_arcsec": [0.5]}
            ),
            matches,
        )
        with pytest.raises(FileNotFoundError, match="Matches file not found"):
            pleiades.merge_match_to_catalog(
                "/nonexistent/m.parquet", cat, tmp_path / "out.parquet"
            )
        with pytest.raises(FileNotFoundError, match="Catalog not found"):
            pleiades.merge_match_to_catalog(
                matches, "/nonexistent/c.parquet", tmp_path / "out.parquet"
            )

    def test_merge_match_to_catalog_how_non_left_warns(self, tmp_path: Path) -> None:
        """Passing how other than 'left' emits UserWarning and still performs left merge."""
        matches = tmp_path / "m.parquet"
        catalog = tmp_path / "cat.parquet"
        out = tmp_path / "out.parquet"
        pq.write_table(
            pa.table(
                {"source_id": [1], "object_id": ["B1"], "separation_arcsec": [0.5]}
            ),
            matches,
        )
        pq.write_table(
            pa.table({"source_id": [1], "ra": [0.0], "dec": [0.0]}),
            catalog,
        )
        with pytest.warns(UserWarning, match="only supports how='left'"):
            pleiades.merge_match_to_catalog(matches, catalog, out, how="inner")
        t = pq.read_table(out)
        assert t.num_rows == 1
        assert "separation_arcsec" in t.column_names


@pytest.mark.unit
class TestFilterMatchesByRadius:
    """Tests for filter_matches_by_radius()."""

    def test_filter_matches_by_radius(self, tmp_path: Path) -> None:
        """Filter keeps only rows with separation <= max_radius."""
        inp = tmp_path / "in.parquet"
        out = tmp_path / "out.parquet"
        pq.write_table(
            pa.table(
                {
                    "source_id": [1, 2, 3],
                    "object_id": ["B1", "B2", "B3"],
                    "separation_arcsec": [0.5, 1.5, 2.5],
                }
            ),
            inp,
        )
        n = pleiades.filter_matches_by_radius(inp, 2.0, out)
        assert n == 2
        t = pq.read_table(out)
        assert t.num_rows == 2
        assert max(t.column("separation_arcsec")[i].as_py() for i in range(2)) <= 2.0

    def test_filter_matches_by_radius_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when matches file missing."""
        with pytest.raises(FileNotFoundError, match="Matches file not found"):
            pleiades.filter_matches_by_radius(
                "/nonexistent/m.parquet", 1.0, tmp_path / "out.parquet"
            )


@pytest.mark.unit
class TestMultiRadiusCrossMatch:
    """Tests for multi_radius_cross_match()."""

    def test_multi_radius_cross_match_writes_per_radius(self, tmp_path: Path) -> None:
        """Produces one file per radius and returns path map."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        result = pleiades.multi_radius_cross_match(
            catalog_a=fixtures / "catalog_a_small.parquet",
            catalog_b=fixtures / "catalog_b_small.parquet",
            radii_arcsec=[1.0, 2.0, 5.0],
            output_dir=tmp_path,
        )
        assert len(result) == 3
        assert 1.0 in result
        assert 2.0 in result
        assert 5.0 in result
        for r, path in result.items():
            assert Path(path).is_file()
            t = pq.read_table(path)
            if t.num_rows > 0:
                assert (
                    max(
                        t.column("separation_arcsec")[i].as_py()
                        for i in range(t.num_rows)
                    )
                    <= r
                )
        combined = tmp_path / "matches_max_radius.parquet"
        assert combined.is_file()
        n_max = pq.read_table(combined).num_rows
        n_1 = pq.read_table(result[1.0]).num_rows
        n_2 = pq.read_table(result[2.0]).num_rows
        assert n_1 <= n_2 <= n_max
