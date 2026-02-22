"""Unit tests for pleiades.models."""

from __future__ import annotations

import pytest

from pleiades.models import CrossMatchResult, MatchSummary


@pytest.mark.unit
class TestCrossMatchResult:
    """Tests for CrossMatchResult model."""

    def test_create_and_access(self) -> None:
        """Fields are readable and frozen."""
        r = CrossMatchResult(
            output_path="/out.parquet",
            rows_a_read=100,
            rows_b_read=200,
            matches_count=10,
            chunks_processed=1,
            time_seconds=1.5,
        )
        assert r.output_path == "/out.parquet"
        assert r.rows_a_read == 100
        assert r.matches_count == 10
        assert r.time_seconds == 1.5

    def test_frozen_raises_on_assignment(self) -> None:
        """CrossMatchResult is immutable."""
        r = CrossMatchResult(
            output_path="/out.parquet",
            rows_a_read=0,
            rows_b_read=0,
            matches_count=0,
            chunks_processed=0,
            time_seconds=0.0,
        )
        with pytest.raises(Exception):  # ValidationError from pydantic
            setattr(r, "matches_count", 1)


@pytest.mark.unit
class TestMatchSummary:
    """Tests for MatchSummary model."""

    def test_create_with_median(self) -> None:
        """MatchSummary with median."""
        s = MatchSummary(
            num_matches=5,
            num_unique_id_a=3,
            num_unique_id_b=4,
            separation_arcsec_min=0.1,
            separation_arcsec_max=2.0,
            separation_arcsec_mean=1.0,
            separation_arcsec_median=1.0,
            id_a_column="source_id",
            id_b_column="object_id",
        )
        assert s.separation_arcsec_median == 1.0
        assert s.id_a_column == "source_id"

    def test_create_median_none(self) -> None:
        """MatchSummary with median=None (empty file)."""
        s = MatchSummary(
            num_matches=0,
            num_unique_id_a=0,
            num_unique_id_b=0,
            separation_arcsec_min=0.0,
            separation_arcsec_max=0.0,
            separation_arcsec_mean=0.0,
            separation_arcsec_median=None,
            id_a_column="id_a",
            id_b_column="id_b",
        )
        assert s.separation_arcsec_median is None

    def test_to_dict(self) -> None:
        """to_dict returns plain dict with all fields."""
        s = MatchSummary(
            num_matches=2,
            num_unique_id_a=2,
            num_unique_id_b=2,
            separation_arcsec_min=0.5,
            separation_arcsec_max=1.0,
            separation_arcsec_mean=0.75,
            separation_arcsec_median=0.75,
            id_a_column="source_id",
            id_b_column="object_id",
        )
        d = s.to_dict()
        assert isinstance(d, dict)
        assert d["num_matches"] == 2
        assert d["separation_arcsec_median"] == 0.75
        assert d["id_a_column"] == "source_id"
