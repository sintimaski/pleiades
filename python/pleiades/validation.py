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
    path: str | Path,
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

    Args:
        path: Path to the Parquet file (file must exist and be a regular file).
        ra_col: Name of the RA column (default "ra").
        dec_col: Name of the Dec column (default "dec").
        id_col: If set, this column must exist in the schema.
        must_have_id: If True, at least one non-ra/dec column must exist.

    Returns:
        The Parquet schema if valid.

    Raises:
        FileNotFoundError: If path does not exist or is not a regular file.
        CatalogValidationError: If schema is invalid.
    """
    path = Path(path)
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


def validate_prepartitioned_dir(path: str | Path) -> tuple[int, pa.Schema]:
    """
    Validate a directory of HEALPix shard Parquet files (shard_0000.parquet, ...).

    Args:
        path: Path to the directory containing shard_*.parquet files.

    Returns:
        (n_shards, schema) where schema is from the first shard.

    Raises:
        CatalogValidationError: If directory is invalid or empty.
    """
    path = Path(path)
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


def validate_output_path(
    path: str | Path,
    *,
    path_type: str = "file",
    base_dir: str | Path | None = None,
) -> Path:
    """
    Validate an output path for safety (e.g. path traversal).

    Resolves the path and optionally ensures it is under base_dir.
    Use base_dir in production to restrict writes to a known directory.

    Args:
        path: Intended output path (file or directory).
        path_type: "file" or "dir" — for file, parent dir is checked.
        base_dir: If set, resolved path must be under this directory.

    Returns:
        Resolved Path (file or dir, parent created later by caller).

    Raises:
        CatalogValidationError: If path is invalid or escapes base_dir.
    """
    p = Path(path).resolve()
    if base_dir is not None:
        base = Path(base_dir).resolve()
        try:
            p_relative = p.relative_to(base)
        except ValueError:
            raise CatalogValidationError(
                f"Output path {p} is not under base directory {base}. "
                "Restrict output to a known directory in production."
            ) from None
        if ".." in p_relative.parts:
            raise CatalogValidationError(
                f"Output path resolves outside base directory: {p}"
            )
    if path_type == "file":
        parent = p.parent
        if parent != p and not parent.exists():
            # Allow creation later; just ensure parent path is valid
            pass
    return p
