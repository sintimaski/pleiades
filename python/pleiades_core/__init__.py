"""Native Pleiades engine (Rust extension).

This package is implemented in Rust and built by maturin. The compiled
extension is loaded as the pleiades_core submodule.
"""

from pleiades_core.pleiades_core import cross_match, has_wgpu_feature

try:
    from pleiades_core.pleiades_core import partition_catalog
except AttributeError:
    partition_catalog = None  # type: ignore[assignment]

try:
    from pleiades_core.pleiades_core import cone_search
except AttributeError:
    cone_search = None  # type: ignore[assignment]

try:
    from pleiades_core.pleiades_core import batch_cone_search
except AttributeError:
    batch_cone_search = None  # type: ignore[assignment]

try:
    from pleiades_core.pleiades_core import attach_match_coords
except AttributeError:
    attach_match_coords = None  # type: ignore[assignment]

__all__ = [
    "cross_match",
    "has_wgpu_feature",
    "partition_catalog",
    "cone_search",
    "batch_cone_search",
    "attach_match_coords",
]
