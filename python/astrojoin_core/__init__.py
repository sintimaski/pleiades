"""Native AstroJoin engine (Rust extension).

This package is implemented in Rust and built by maturin. The compiled
extension is loaded as the astrojoin_core submodule.
"""

from astrojoin_core.astrojoin_core import cross_match, has_wgpu_feature

__all__ = ["cross_match", "has_wgpu_feature"]
