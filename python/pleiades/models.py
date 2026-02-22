"""Data models for cross-match results and analysis (Pydantic)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CrossMatchResult(BaseModel):
    """Result metadata returned by cross_match()."""

    output_path: str = Field(description="Path where matches Parquet was written.")
    rows_a_read: int = Field(description="Total rows read from catalog A.")
    rows_b_read: int = Field(
        description="Total rows read from catalog B (or from shards)."
    )
    matches_count: int = Field(description="Total number of match pairs written.")
    chunks_processed: int = Field(description="Number of A chunks processed.")
    time_seconds: float = Field(description="Wall-clock time for the run in seconds.")

    model_config = {"frozen": True}


class MatchSummary(BaseModel):
    """Summary statistics for a matches Parquet file (from summarize_matches())."""

    num_matches: int = Field(description="Total number of match rows.")
    num_unique_id_a: int = Field(description="Number of unique id_a values.")
    num_unique_id_b: int = Field(description="Number of unique id_b values.")
    separation_arcsec_min: float = Field(description="Minimum separation in arcsec.")
    separation_arcsec_max: float = Field(description="Maximum separation in arcsec.")
    separation_arcsec_mean: float = Field(description="Mean separation in arcsec.")
    separation_arcsec_median: float | None = Field(
        default=None, description="Median separation in arcsec (if computable)."
    )
    id_a_column: str = Field(description="Name of the id_a column.")
    id_b_column: str = Field(description="Name of the id_b column.")

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, Any]:
        """Return summary as a plain dict (e.g. for logging)."""
        return self.model_dump()
