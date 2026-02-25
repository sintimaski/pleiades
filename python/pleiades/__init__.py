"""Pleiades: out-of-core spatial cross-matcher for astronomical catalogs.

Use cross_match() to join two Parquet catalogs by angular distance
(e.g. Gaia vs SDSS) without loading full datasets into RAM.
"""

from importlib.metadata import version

__version__ = version("pleiades")

from pleiades.analysis import (
    attach_match_coords,
    filter_matches_by_radius,
    match_quality_summary,
    match_stats,
    merge_match_to_catalog,
    multi_radius_cross_match,
    summarize_matches,
)
from pleiades.cone import batch_cone_search, cone_search
from pleiades.cross_match import (
    cross_match,
    cross_match_iter,
    partition_catalog,
    suggest_healpix_depth,
)
from pleiades.models import CrossMatchResult, MatchSummary
from pleiades.validation import CatalogValidationError

__all__ = [
    "__version__",
    "attach_match_coords",
    "batch_cone_search",
    "CatalogValidationError",
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
    "suggest_healpix_depth",
    "summarize_matches",
]
