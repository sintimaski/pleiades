"""Environment and runtime configuration for Pleiades.

All env vars are optional. Used by the Rust engine and Python logging.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# Environment variables (documented in one place)
# -----------------------------------------------------------------------------
# PLEIADES_VERBOSE: set to 1 or true for INFO-level Python logging and
#   Rust engine timing (stderr). Default: unset (WARNING, no timing).
# PLEIADES_PROFILE: set to 1 for Rust engine profiling/timing output.
# PLEIADES_JOIN_STRATEGY: Rust join strategy (e.g. "matrix", "nested").
# PLEIADES_GPU: set to "wgpu" to enable GPU-accelerated join (if built with wgpu).
# PLEIADES_GPU_MIN_PAIRS: minimum pair count to use GPU (Rust).
# PLEIADES_OUTPUT_BASE_DIR: if set, output_path and output_dir must resolve
#   under this directory (production safety). Not enforced by default.
# -----------------------------------------------------------------------------


def env_bool(name: str, default: bool = False) -> bool:
    """Return True if env var is set to 1, true, or yes (case-insensitive)."""
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes") if val else default


def env_int(name: str, default: int | None = None) -> int | None:
    """Return env var as int, or default if unset/invalid."""
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_verbose() -> bool:
    """True if PLEIADES_VERBOSE is set (Python logging + Rust timing)."""
    return env_bool("PLEIADES_VERBOSE", False)


def get_output_base_dir() -> str | None:
    """
    If set, output paths should resolve under this directory.

    Use in production to restrict where matches and shards are written.
    Pass to validate_output_path(..., base_dir=...) when validating paths.
    """
    val = os.environ.get("PLEIADES_OUTPUT_BASE_DIR", "").strip()
    return val or None
