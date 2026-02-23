"""Analysis helpers for match outputs: summarize, merge to catalog, filter by radius."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pleiades.models import MatchSummary


def match_stats(
    matches_path: str | Path,
    *,
    separation_percentiles: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute rich statistics for a matches Parquet file.

    Returns a dict with num_matches, num_unique_id_a, num_unique_id_b,
    separation min/max/mean/median, optional percentiles (e.g. [25, 50, 75]),
    and id_a_column, id_b_column.
    """
    path = Path(matches_path)
    if not path.is_file():
        raise FileNotFoundError(f"Matches file not found: {path}")
    table = pq.read_table(path)
    names = table.column_names
    id_a_col = next((n for n in names if n != "separation_arcsec"), "id_a")
    id_b_col = next(
        (n for n in names if n != "separation_arcsec" and n != id_a_col),
        "id_b",
    )
    sep = table.column("separation_arcsec")
    sep_np = sep.to_numpy()
    n_unique_a = len(pc.unique(table.column(id_a_col)))
    n_unique_b = len(pc.unique(table.column(id_b_col)))
    out: dict[str, Any] = {
        "num_matches": table.num_rows,
        "num_unique_id_a": n_unique_a,
        "num_unique_id_b": n_unique_b,
        "separation_arcsec_min": float(np.min(sep_np)) if len(sep_np) else 0.0,
        "separation_arcsec_max": float(np.max(sep_np)) if len(sep_np) else 0.0,
        "separation_arcsec_mean": float(np.mean(sep_np)) if len(sep_np) else 0.0,
        "separation_arcsec_median": float(np.median(sep_np)) if len(sep_np) else None,
        "id_a_column": id_a_col,
        "id_b_column": id_b_col,
    }
    if separation_percentiles and len(sep_np) > 0:
        out["separation_percentiles"] = {
            str(p): float(np.percentile(sep_np, p)) for p in separation_percentiles
        }
    return out


def match_quality_summary(
    matches_path: str | Path,
    rows_a: int,
    rows_b: int,
) -> dict[str, float]:
    """
    Summarize match quality: fraction of catalog A and B rows that have at least one match.

    Returns dict with fraction_id_a_matched, fraction_id_b_matched, and num_matches.
    """
    path = Path(matches_path)
    if not path.is_file():
        raise FileNotFoundError(f"Matches file not found: {path}")
    table = pq.read_table(path)
    if table.num_rows == 0:
        return {
            "fraction_id_a_matched": 0.0,
            "fraction_id_b_matched": 0.0,
            "num_matches": 0,
        }
    names = table.column_names
    id_a_col = next((n for n in names if n != "separation_arcsec"), "id_a")
    id_b_col = next(
        (n for n in names if n != "separation_arcsec" and n != id_a_col),
        "id_b",
    )
    n_unique_a = len(pc.unique(table.column(id_a_col)))
    n_unique_b = len(pc.unique(table.column(id_b_col)))
    frac_a = n_unique_a / rows_a if rows_a > 0 else 0.0
    frac_b = n_unique_b / rows_b if rows_b > 0 else 0.0
    return {
        "fraction_id_a_matched": frac_a,
        "fraction_id_b_matched": frac_b,
        "num_matches": table.num_rows,
    }


def summarize_matches(matches_path: str | Path) -> MatchSummary:
    """
    Compute summary statistics for a matches Parquet file.

    Returns counts, unique id_a/id_b, and separation min/max/mean/median.
    """
    path = Path(matches_path)
    if not path.is_file():
        raise FileNotFoundError(f"Matches file not found: {path}")

    table = pq.read_table(path)
    if table.num_rows == 0:
        names = table.column_names
        id_a_col = next((n for n in names if n != "separation_arcsec"), "id_a")
        id_b_col = next(
            (n for n in names if n != "separation_arcsec" and n != id_a_col),
            "id_b",
        )
        return MatchSummary(
            num_matches=0,
            num_unique_id_a=0,
            num_unique_id_b=0,
            separation_arcsec_min=0.0,
            separation_arcsec_max=0.0,
            separation_arcsec_mean=0.0,
            separation_arcsec_median=None,
            id_a_column=id_a_col,
            id_b_column=id_b_col,
        )

    sep = table.column("separation_arcsec")
    sep_np = sep.to_numpy()
    names = table.column_names
    id_a_col = next((n for n in names if n != "separation_arcsec"), "id_a")
    id_b_col = next(
        (n for n in names if n != "separation_arcsec" and n != id_a_col),
        "id_b",
    )
    id_a_arr = table.column(id_a_col)
    id_b_arr = table.column(id_b_col)

    n_unique_a = len(pc.unique(id_a_arr))
    n_unique_b = len(pc.unique(id_b_arr))

    return MatchSummary(
        num_matches=table.num_rows,
        num_unique_id_a=n_unique_a,
        num_unique_id_b=n_unique_b,
        separation_arcsec_min=float(np.min(sep_np)),
        separation_arcsec_max=float(np.max(sep_np)),
        separation_arcsec_mean=float(np.mean(sep_np)),
        separation_arcsec_median=float(np.median(sep_np)),
        id_a_column=id_a_col,
        id_b_column=id_b_col,
    )


def merge_match_to_catalog(
    matches_path: str | Path,
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    how: str = "left",
    catalog_side: str = "a",
    id_col_catalog: str | None = None,
) -> None:
    """
    Join match results onto a catalog by ID.

    Merges the matches table (id_a, id_b, separation_arcsec) with the catalog
    so each catalog row gets match columns (id_b or id_a, separation_arcsec).
    catalog_side="a" means catalog rows are identified by id_a (merge on
    catalog.id == matches.id_a); catalog_side="b" uses id_b.

    Only a left-join style is implemented: one row per catalog row, with the
    first match (by match key) attached. The ``how`` parameter is accepted
    for API compatibility but only "left" is supported; "inner" and "right"
    are ignored.

    Args:
        matches_path: Path to matches Parquet (id_a, id_b, separation_arcsec).
        catalog_path: Path to catalog Parquet (must have an ID column).
        output_path: Where to write the merged Parquet.
        how: Join type; only "left" is supported (other values are ignored).
        catalog_side: "a" or "b" — which match column corresponds to catalog IDs.
        id_col_catalog: Catalog ID column name; inferred if None.
    """
    if how != "left":
        import warnings

        warnings.warn(
            f"merge_match_to_catalog only supports how='left'; how={how!r} is ignored.",
            UserWarning,
            stacklevel=2,
        )
    path_m = Path(matches_path)
    path_c = Path(catalog_path)
    path_out = Path(output_path)
    if not path_m.is_file():
        raise FileNotFoundError(f"Matches file not found: {path_m}")
    if not path_c.is_file():
        raise FileNotFoundError(f"Catalog not found: {path_c}")

    matches = pq.read_table(path_m)
    catalog = pq.read_table(path_c)

    id_a_col = next(
        (n for n in matches.column_names if n != "separation_arcsec"), "id_a"
    )
    id_b_col = next(
        (n for n in matches.column_names if n != "separation_arcsec" and n != id_a_col),
        "id_b",
    )

    if catalog_side == "a":
        match_id_col = id_a_col
        other_col = id_b_col
    else:
        match_id_col = id_b_col
        other_col = id_a_col

    if id_col_catalog is None:
        for name in catalog.column_names:
            if name.lower() not in ("ra", "dec"):
                id_col_catalog = name
                break
        else:
            id_col_catalog = "id"
    if id_col_catalog not in catalog.column_names:
        raise ValueError(
            f"Catalog ID column '{id_col_catalog}' not in catalog columns: "
            f"{catalog.column_names}"
        )

    # Build match key -> (other_id, separation) for first match per key
    # (or keep all and do a proper join; for simplicity we do one row per catalog row)
    match_keys = matches.column(match_id_col)
    match_other = matches.column(other_col)
    match_sep = matches.column("separation_arcsec")
    key_to_other: dict[Any, tuple[Any, float]] = {}
    for i in range(matches.num_rows):
        k = match_keys[i].as_py()
        if k not in key_to_other:
            key_to_other[k] = (match_other[i].as_py(), match_sep[i].as_py())
    del matches

    catalog_ids = catalog.column(id_col_catalog)
    other_vals: list[Any] = []
    sep_vals: list[float | None] = []
    for i in range(catalog.num_rows):
        cid = catalog_ids[i].as_py()
        pair = key_to_other.get(cid)
        if pair is not None:
            other_vals.append(pair[0])
            sep_vals.append(pair[1])
        else:
            other_vals.append(None)
            sep_vals.append(None)

    out_columns = [catalog.column(name) for name in catalog.column_names]
    out_names = list(catalog.column_names)
    out_columns.append(pa.array(other_vals))
    out_names.append(f"match_{other_col}")
    out_columns.append(pa.array(sep_vals))
    out_names.append("separation_arcsec")
    out_table = pa.table(dict(zip(out_names, out_columns, strict=True)))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, path_out)


def filter_matches_by_radius(
    matches_path: str | Path,
    max_radius_arcsec: float,
    output_path: str | Path,
) -> int:
    """
    Write a subset of matches with separation_arcsec <= max_radius_arcsec.

    Returns the number of rows written.
    """
    path = Path(matches_path)
    path_out = Path(output_path)
    if not path.is_file():
        raise FileNotFoundError(f"Matches file not found: {path}")

    table = pq.read_table(path)
    sep = table.column("separation_arcsec")
    mask = pc.less_equal(sep, max_radius_arcsec)
    filtered = table.filter(mask)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, path_out)
    return int(filtered.num_rows)


def attach_match_coords(
    matches_path: str | Path,
    catalog_a_path: str | Path,
    catalog_b_path: str | Path,
    output_path: str | Path,
    *,
    id_col_a: str | None = None,
    id_col_b: str | None = None,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> None:
    """
    Add ra_a, dec_a, ra_b, dec_b columns to a matches Parquet from the two catalogs.

    Reads matches and both catalogs, joins on id_a/id_b to attach coordinates,
    and writes the result to output_path. Useful when include_coords=True was
    not used during cross_match.
    """
    path_m = Path(matches_path)
    path_a = Path(catalog_a_path)
    path_b = Path(catalog_b_path)
    path_out = Path(output_path)
    if not path_m.is_file():
        raise FileNotFoundError(f"Matches file not found: {path_m}")
    if not path_a.is_file():
        raise FileNotFoundError(f"Catalog A not found: {path_a}")
    if not path_b.is_file():
        raise FileNotFoundError(f"Catalog B not found: {path_b}")
    matches = pq.read_table(path_m)
    catalog_a = pq.read_table(path_a)
    catalog_b = pq.read_table(path_b)
    names = matches.column_names
    id_a_name = next((n for n in names if n != "separation_arcsec"), "id_a")
    id_b_name = next(
        (n for n in names if n != "separation_arcsec" and n != id_a_name),
        "id_b",
    )
    if id_col_a is None:
        id_col_a = next(
            (n for n in catalog_a.column_names if n.lower() not in ("ra", "dec")),
            "id",
        )
    if id_col_b is None:
        id_col_b = next(
            (n for n in catalog_b.column_names if n.lower() not in ("ra", "dec")),
            "id",
        )

    def build_id_to_ra_dec(
        tbl: pa.Table, id_name: str
    ) -> dict[Any, tuple[float, float]]:
        out: dict[Any, tuple[float, float]] = {}
        id_col = tbl.column(id_name)
        ra_col_arr = tbl.column(ra_col)
        dec_col_arr = tbl.column(dec_col)
        for i in range(tbl.num_rows):
            k = id_col[i].as_py() if hasattr(id_col[i], "as_py") else id_col[i]
            out[k] = (float(ra_col_arr[i].as_py()), float(dec_col_arr[i].as_py()))
        return out

    a_map = build_id_to_ra_dec(catalog_a, id_col_a or "id")
    b_map = build_id_to_ra_dec(catalog_b, id_col_b or "id")
    id_a_arr = matches.column(id_a_name)
    id_b_arr = matches.column(id_b_name)
    ra_a_list: list[float | None] = []
    dec_a_list: list[float | None] = []
    ra_b_list: list[float | None] = []
    dec_b_list: list[float | None] = []
    for i in range(matches.num_rows):
        ka = id_a_arr[i].as_py() if hasattr(id_a_arr[i], "as_py") else id_a_arr[i]
        kb = id_b_arr[i].as_py() if hasattr(id_b_arr[i], "as_py") else id_b_arr[i]
        pa_coord = a_map.get(ka)
        pb_coord = b_map.get(kb)
        if pa_coord:
            ra_a_list.append(pa_coord[0])
            dec_a_list.append(pa_coord[1])
        else:
            ra_a_list.append(None)
            dec_a_list.append(None)
        if pb_coord:
            ra_b_list.append(pb_coord[0])
            dec_b_list.append(pb_coord[1])
        else:
            ra_b_list.append(None)
            dec_b_list.append(None)
    new_columns = list(matches.columns)
    new_names = list(matches.column_names)
    new_columns.extend(
        [
            pa.array(ra_a_list),
            pa.array(dec_a_list),
            pa.array(ra_b_list),
            pa.array(dec_b_list),
        ]
    )
    new_names.extend(["ra_a", "dec_a", "ra_b", "dec_b"])
    out_table = pa.table(dict(zip(new_names, new_columns, strict=True)))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, path_out)


def multi_radius_cross_match(
    catalog_a: str | Path,
    catalog_b: str | Path,
    radii_arcsec: list[float],
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[float, str]:
    """
    Run cross-match once with max(radii), then filter to each radius.

    Writes one Parquet per radius to output_dir (e.g. matches_1.0.parquet).
    Returns a mapping radius_arcsec -> output file path.
    """
    from pleiades.cross_match import cross_match

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_radius = max(radii_arcsec)
    combined = output_dir / "matches_max_radius.parquet"
    cross_match(
        catalog_a,
        catalog_b,
        max_radius,
        combined,
        **kwargs,
    )
    result: dict[float, str] = {}
    for r in radii_arcsec:
        out = output_dir / f"matches_{r}.parquet"
        filter_matches_by_radius(combined, r, out)
        result[r] = str(out)
    return result
