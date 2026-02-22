"""AstroJoin: out-of-core spatial cross-matcher for astronomical catalogs.

Use cross_match() to join two Parquet catalogs by angular distance
(e.g. Gaia vs SDSS) without loading full datasets into RAM.
"""

from astrojoin.analysis import (
    attach_match_coords,
    filter_matches_by_radius,
    match_quality_summary,
    match_stats,
    merge_match_to_catalog,
    multi_radius_cross_match,
    summarize_matches,
)
from astrojoin.cone import batch_cone_search, cone_search
from astrojoin.cross_match import cross_match, cross_match_iter, partition_catalog
from astrojoin.models import CrossMatchResult, MatchSummary

__all__ = [
    "attach_match_coords",
    "batch_cone_search",
    "cone_search",
    "cross_match",
    "cross_match_iter",
    "CrossMatchResult",
    "filter_matches_by_radius",
    "match_quality_summary",
    "match_stats",
    "merge_match_to_catalog",
    "MatchSummary",
    "multi_radius_cross_match",
    "partition_catalog",
    "summarize_matches",
]
