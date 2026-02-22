"""Schema validation for Parquet catalogs before cross-match."""

from __future__ import annotations

import difflib
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class CatalogValidationError(ValueError):
    """Raised when a catalog fails schema validation."""

    pass


def _suggest_column(name: str, available: list[str], cutoff: float = 0.5) -> str | None:
    """Return a close match from available names if any, else None."""
    if not available:
        return None
    matches = difflib.get_close_matches(name, available, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def validate_cross_match_args(
    radius_arcsec: float,
    n_nearest: int | None = None,
    depth: int = 8,
    n_shards: int = 512,
) -> None:
    """
    Validate cross_match parameters. Raises CatalogValidationError if invalid.
    """
    if math.isnan(radius_arcsec):
        raise CatalogValidationError("radius_arcsec must not be NaN")
    if not math.isfinite(radius_arcsec):
        raise CatalogValidationError(
            f"radius_arcsec must be finite, got {radius_arcsec!r}"
        )
    if radius_arcsec <= 0:
        raise CatalogValidationError(
            f"radius_arcsec must be positive, got {radius_arcsec!r}"
        )
    if n_nearest is not None and n_nearest < 1:
        raise CatalogValidationError(
            f"n_nearest must be >= 1 when set, got {n_nearest!r}"
        )
    if depth < 0 or depth > 15:
        raise CatalogValidationError(
            f"depth must be between 0 and 15 (HEALPix), got {depth!r}"
        )
    if n_shards < 1:
        raise CatalogValidationError(f"n_shards must be >= 1, got {n_shards!r}")


def validate_catalog_schema(
    path: Path,
    *,
    ra_col: str = "ra",
    dec_col: str = "dec",
    id_col: str | None = None,
    must_have_id: bool = False,
) -> pa.Schema:
    """
    Validate that a Parquet file has required columns for cross-match.

    Checks that ra_col and dec_col exist and are numeric (float or int).
    If id_col is provided, checks it exists. If must_have_id is True,
    ensures at least one non-ra/dec column exists.

    Returns:
        The Parquet schema if valid.

    Raises:
        FileNotFoundError: If path does not exist.
        CatalogValidationError: If schema is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Catalog not found: {path}")

    schema = pq.read_schema(path)
    names_lower = {name.lower(): name for name in schema.names}

    def resolve(name: str) -> str:
        key = name.lower()
        if key not in names_lower:
            suggestion = _suggest_column(name, schema.names)
            msg = (
                f"Catalog {path}: required column '{name}' not found. "
                f"Available columns: {list(schema.names)}"
            )
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            raise CatalogValidationError(msg)
        return str(names_lower[key])

    for col_name, _key in [(ra_col, "ra"), (dec_col, "dec")]:
        resolved = resolve(col_name)
        field = schema.field(resolved)
        if not pa.types.is_floating(field.type) and not pa.types.is_integer(field.type):
            raise CatalogValidationError(
                f"Catalog {path}: column '{resolved}' must be numeric (float or int), "
                f"got {field.type}. Use ra_col/dec_col if your coordinates use different names."
            )

    if id_col is not None:
        resolve(id_col)

    if must_have_id:
        non_radec = [
            n
            for n in schema.names
            if n.lower() not in (ra_col.lower(), dec_col.lower())
        ]
        if not non_radec:
            raise CatalogValidationError(
                f"Catalog {path}: at least one non-ra/dec column required for IDs. "
                f"Columns: {list(schema.names)}"
            )

    return schema


def validate_prepartitioned_dir(path: Path) -> tuple[int, pa.Schema]:
    """
    Validate a directory of HEALPix shard Parquet files (shard_0000.parquet, ...).

    Returns:
        (n_shards, schema) where schema is from the first shard.

    Raises:
        CatalogValidationError: If directory is invalid or empty.
    """
    if not path.is_dir():
        raise CatalogValidationError(
            f"Pre-partitioned B path is not a directory: {path}"
        )

    shard_files = sorted(path.glob("shard_*.parquet"))
    if not shard_files:
        raise CatalogValidationError(
            f"Pre-partitioned B directory has no shard_*.parquet files: {path}"
        )

    required = {"pixel_id", "id_b", "ra", "dec"}
    first = pq.read_schema(shard_files[0])
    first_set = set(first.names)
    missing = required - first_set
    if missing:
        raise CatalogValidationError(
            f"Pre-partitioned shard must have columns {required}, got {first.names}"
        )

    n_shards = len(shard_files)
    return n_shards, first
