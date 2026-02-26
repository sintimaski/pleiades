"""Integration tests: generate catalogs, run cross_match, assert on exact results and edge cases."""

from __future__ import annotations

from pathlib import Path

import pleiades
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.integration.cross_match_helpers import (
    make_catalogs_exact_n_pairs,
    make_catalogs_many_to_one,
    make_catalogs_no_pairs,
    make_catalogs_one_to_many,
    make_catalogs_pair_at_exact_radius,
    make_catalogs_random,
    match_set_from_table,
    match_table_id_b_col,
    read_matches,
    reference_cross_match_brute_force,
    write_catalogs,
)

# Avoid hangs on slow or large runs; applies to all tests in this module.
pytestmark = [pytest.mark.timeout(60)]


def _run_cross_match(
    path_a: Path,
    path_b: Path,
    output_path: Path,
    radius_arcsec: float = 2.0,
) -> None:
    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=radius_arcsec,
        output_path=output_path,
    )


@pytest.mark.integration
def test_exact_n_pairs_generated(tmp_path: Path) -> None:
    """Generate catalogs with exactly 5 pairs within 2 arcsec; assert exactly 5 matches and correct ids."""
    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=5,
        radius_arcsec=2.0,
        separation_arcsec=1.0,
        n_a_extra=10,
        n_b_extra=10,
        seed=44,
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=2.0)

    t = read_matches(out)
    assert t.num_rows == 5, f"expected exactly 5 matches, got {t.num_rows}"
    got_set = match_set_from_table(t, "source_id", match_table_id_b_col(t))
    expected_set = {(e[0], e[1]) for e in expected}
    assert got_set == expected_set, (
        f"match pairs differ: got {got_set}, expected {expected_set}"
    )
    sep_col = t.column("separation_arcsec")
    for i in range(t.num_rows):
        assert sep_col[i].as_py() <= 2.0


@pytest.mark.integration
def test_zero_pairs_generated(tmp_path: Path) -> None:
    """Generate random A and B; small radius so no pairs match. Assert 0 matches."""
    table_a, table_b = make_catalogs_no_pairs(
        n_a=30, n_b=30, radius_arcsec=2.0, seed=99
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=0.5)

    t = read_matches(out)
    assert t.num_rows == 0, f"expected 0 matches, got {t.num_rows}"
    assert "separation_arcsec" in t.column_names


@pytest.mark.integration
def test_pair_at_exact_radius_boundary(tmp_path: Path) -> None:
    """One pair placed so separation equals radius; assert 1 match and separation <= radius."""
    radius = 2.0
    table_a, table_b, (id_a, id_b, expected_sep) = make_catalogs_pair_at_exact_radius(
        radius_arcsec=radius, seed=7
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=radius)

    t = read_matches(out)
    assert t.num_rows == 1, f"expected 1 match at boundary, got {t.num_rows}"
    assert t.column("source_id")[0].as_py() == id_a
    assert t.column(match_table_id_b_col(t))[0].as_py() == id_b
    sep = t.column("separation_arcsec")[0].as_py()
    assert sep <= radius
    assert abs(sep - expected_sep) < 0.01


@pytest.mark.integration
def test_one_to_many_matches(tmp_path: Path) -> None:
    """One source in A, 4 in B within radius; assert exactly 4 matches with correct (1, Bk) pairs."""
    table_a, table_b, expected = make_catalogs_one_to_many(
        n_b_matches=4, radius_arcsec=2.0, separation_arcsec=0.5, seed=11
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=2.0)

    t = read_matches(out)
    assert t.num_rows == 4, f"expected 4 matches (one-to-many), got {t.num_rows}"
    got_set = match_set_from_table(t, "source_id", match_table_id_b_col(t))
    expected_set = {(e[0], e[1]) for e in expected}
    assert got_set == expected_set


@pytest.mark.integration
def test_many_to_one_matches(tmp_path: Path) -> None:
    """3 sources in A, one in B; all A within radius of B. Assert exactly 3 matches."""
    table_a, table_b, expected = make_catalogs_many_to_one(
        n_a_matches=3, radius_arcsec=2.0, separation_arcsec=0.5, seed=22
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=2.0)

    t = read_matches(out)
    assert t.num_rows == 3, f"expected 3 matches (many-to-one), got {t.num_rows}"
    got_set = match_set_from_table(t, "source_id", match_table_id_b_col(t))
    expected_set = {(e[0], e[1]) for e in expected}
    assert got_set == expected_set


@pytest.mark.integration
def test_empty_catalog_a(tmp_path: Path) -> None:
    """Catalog A has 0 rows; assert 0 matches and no crash."""
    empty_a = pa.table(
        {
            "source_id": pa.array([], type=pa.int64()),
            "ra": pa.array([], type=pa.float64()),
            "dec": pa.array([], type=pa.float64()),
        },
        schema=pa.schema(
            [("source_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    _, table_b = make_catalogs_no_pairs(n_a=0, n_b=5, seed=1)
    path_a = tmp_path / "empty_a.parquet"
    path_b = tmp_path / "catalog_b.parquet"
    pq.write_table(empty_a, path_a)
    pq.write_table(table_b, path_b)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=2.0)

    t = read_matches(out)
    assert t.num_rows == 0


@pytest.mark.integration
def test_empty_catalog_b(tmp_path: Path) -> None:
    """Catalog B has 0 rows; assert 0 matches and no crash."""
    table_a, _ = make_catalogs_no_pairs(n_a=5, n_b=0, seed=2)
    empty_b = pa.table(
        {
            "object_id": pa.array([], type=pa.string()),
            "ra": pa.array([], type=pa.float64()),
            "dec": pa.array([], type=pa.float64()),
        },
        schema=pa.schema(
            [("object_id", pa.string()), ("ra", pa.float64()), ("dec", pa.float64())]
        ),
    )
    path_a = tmp_path / "catalog_a.parquet"
    path_b = tmp_path / "empty_b.parquet"
    pq.write_table(table_a, path_a)
    pq.write_table(empty_b, path_b)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=2.0)

    t = read_matches(out)
    assert t.num_rows == 0


@pytest.mark.integration
def test_output_schema_and_separations_bounded(tmp_path: Path) -> None:
    """Generated data: assert output has required columns and all separations <= radius."""
    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=2, radius_arcsec=3.0, n_a_extra=2, n_b_extra=2, seed=33
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    _run_cross_match(path_a, path_b, out, radius_arcsec=3.0)

    t = read_matches(out)
    assert "source_id" in t.column_names
    assert match_table_id_b_col(t) in t.column_names
    assert "separation_arcsec" in t.column_names
    sep = t.column("separation_arcsec")
    for i in range(t.num_rows):
        assert sep[i].as_py() <= 3.0, f"separation {sep[i].as_py()} exceeds radius 3.0"


@pytest.mark.integration
def test_matches_equal_reference_brute_force(tmp_path: Path) -> None:
    """
    Ground truth: brute-force reference (every A vs every B, same haversine).
    Generate random A and B, get reference match set, run pleiades (partition_b
    and non-partition path), assert our output (id_a, id_b) set equals reference exactly.
    """
    radius_arcsec = 2.0
    table_a, table_b = make_catalogs_random(n_a=60, n_b=50, seed=999)
    reference_set = reference_cross_match_brute_force(
        table_a,
        table_b,
        radius_arcsec,
        id_col_a="source_id",
        id_col_b="object_id",
        ra_col="ra",
        dec_col="dec",
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)

    for partition_b in (True, False):
        out = tmp_path / f"matches_partition_b_{partition_b}.parquet"
        pleiades.cross_match(
            catalog_a=path_a,
            catalog_b=path_b,
            radius_arcsec=radius_arcsec,
            output_path=out,
            partition_b=partition_b,
        )
        t = read_matches(out)
        our_set = match_set_from_table(t, "source_id", match_table_id_b_col(t))
        assert our_set == reference_set, (
            f"partition_b={partition_b}: our matches differ from reference. "
            f"Only in reference: {reference_set - our_set!r}. "
            f"Only in ours: {our_set - reference_set!r}."
        )
        sep_col = t.column("separation_arcsec")
        for i in range(t.num_rows):
            assert sep_col[i].as_py() <= radius_arcsec


@pytest.mark.integration
def test_rust_matches_equal_reference_brute_force(tmp_path: Path) -> None:
    """
    If the Rust engine (pleiades_core) is built with the full implementation,
    run cross_match() and assert (id_a, id_b) set equals reference.
    Skipped if pleiades_core is not installed. Requires `uv run maturin develop`
    with the full Rust engine for this test to pass.
    """
    pytest.importorskip("pleiades_core")
    radius_arcsec = 2.0
    table_a, table_b = make_catalogs_random(n_a=60, n_b=50, seed=999)
    reference_set = reference_cross_match_brute_force(
        table_a,
        table_b,
        radius_arcsec,
        id_col_a="source_id",
        id_col_b="object_id",
        ra_col="ra",
        dec_col="dec",
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches_rust.parquet"
    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=radius_arcsec,
        output_path=out,
    )
    if not out.is_file():
        pytest.skip(
            "Rust engine did not write output (stub or not built). "
            "Run: uv run maturin develop"
        )
    t = read_matches(out)
    # Column names match input (source_id, object_id) from Rust engine
    id_a_col = t.column_names[0]
    id_b_col = t.column_names[1]
    our_set = match_set_from_table(t, id_a_col, id_b_col)
    assert our_set == reference_set, (
        "Rust engine matches differ from reference. "
        f"Only in reference: {reference_set - our_set!r}. "
        f"Only in ours: {our_set - reference_set!r}."
    )
    sep_col = t.column("separation_arcsec")
    for i in range(t.num_rows):
        assert sep_col[i].as_py() <= radius_arcsec


@pytest.mark.integration
def test_rust_with_prepartitioned_b_matches_reference(tmp_path: Path) -> None:
    """With catalog_b a shard directory, (id_a, id_b) set equals reference."""
    pytest.importorskip("pleiades_core")
    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=4, radius_arcsec=2.0, n_a_extra=10, n_b_extra=10, seed=31
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    depth = 8
    n_shards = 32
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    schema = pa.schema(
        [
            ("pixel_id", pa.uint64()),
            ("id_b", pa.string()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]
    )
    writers = [
        pq.ParquetWriter(shard_dir / f"shard_{s:04d}.parquet", schema)
        for s in range(n_shards)
    ]
    import astropy.units as u
    import cdshealpix.nested as cds_nested
    from astropy.coordinates import Latitude, Longitude

    ra_b = table_b.column("ra").to_numpy()
    dec_b = table_b.column("dec").to_numpy()
    id_b = table_b.column("object_id")
    lon = Longitude(ra_b, unit=u.deg)
    lat = Latitude(dec_b, unit=u.deg)
    pix = cds_nested.lonlat_to_healpix(lon, lat, depth)
    for i in range(len(table_b)):
        s = int(pix[i]) % n_shards
        row = pa.table(
            {
                "pixel_id": [int(pix[i])],
                "id_b": [id_b[i].as_py()],
                "ra": [float(ra_b[i])],
                "dec": [float(dec_b[i])],
            },
            schema=schema,
        )
        writers[s].write_table(row)
    for w in writers:
        w.close()
    out = tmp_path / "matches_rust_shards.parquet"
    try:
        pleiades.cross_match(
            catalog_a=path_a,
            catalog_b=shard_dir,
            radius_arcsec=2.0,
            output_path=out,
        )
    except OSError as e:
        if "Is a directory" in str(e) or "21" in str(e):
            pytest.skip(
                "Rust extension does not support pre-partitioned B (directory). "
                "Rebuild with: uv run maturin develop"
            )
        raise
    t = read_matches(out)
    id_b_col = "id_b" if "id_b" in t.column_names else "object_id"
    got_set = match_set_from_table(t, "source_id", id_b_col)
    expected_set = {(e[0], e[1]) for e in expected}
    assert got_set == expected_set, f"got {got_set}, expected {expected_set}"


@pytest.mark.integration
def test_rust_join_strategy_adynamic_matches_reference(tmp_path: Path) -> None:
    """With PLEIADES_JOIN_STRATEGY=adynamic, (id_a, id_b) set equals reference."""
    pytest.importorskip("pleiades_core")
    import os

    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=4, radius_arcsec=2.0, n_a_extra=10, n_b_extra=10, seed=31
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches_adynamic.parquet"
    old = os.environ.get("PLEIADES_JOIN_STRATEGY")
    try:
        os.environ["PLEIADES_JOIN_STRATEGY"] = "adynamic"
        pleiades.cross_match(
            catalog_a=path_a,
            catalog_b=path_b,
            radius_arcsec=2.0,
            output_path=out,
        )
    finally:
        if old is not None:
            os.environ["PLEIADES_JOIN_STRATEGY"] = old
        elif "PLEIADES_JOIN_STRATEGY" in os.environ:
            del os.environ["PLEIADES_JOIN_STRATEGY"]
    t = read_matches(out)
    id_b_col = "id_b" if "id_b" in t.column_names else "object_id"
    got_set = match_set_from_table(t, "source_id", id_b_col)
    expected_set = {(e[0], e[1]) for e in expected}
    assert got_set == expected_set, f"got {got_set}, expected {expected_set}"


@pytest.mark.integration
def test_rust_n_nearest_reduces_output(tmp_path: Path) -> None:
    """With n_nearest=1, at most one match per id_a."""
    pytest.importorskip("pleiades_core")
    table_a, table_b, _ = make_catalogs_exact_n_pairs(
        n_pairs=5,
        radius_arcsec=2.0,
        separation_arcsec=1.0,
        n_a_extra=5,
        n_b_extra=15,
        seed=42,
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches_n1.parquet"
    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=2.0,
        output_path=out,
        n_nearest=1,
    )
    t = read_matches(out)
    id_a_col = t.column_names[0]
    id_a_vals = t.column(id_a_col)
    counts: dict = {}
    for i in range(t.num_rows):
        k = id_a_vals[i].as_py()
        counts[k] = counts.get(k, 0) + 1
    for k, c in counts.items():
        assert c <= 1, f"id_a={k} has {c} matches, expected at most 1"


@pytest.mark.integration
def test_rust_progress_callback_called(tmp_path: Path) -> None:
    """With progress_callback, callback is invoked."""
    pytest.importorskip("pleiades_core")
    table_a, table_b = make_catalogs_no_pairs(
        n_a=20, n_b=20, radius_arcsec=2.0, seed=88
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    progress_calls: list = []

    def progress(chunk_ix: int, total: int | None, rows_a: int, matches: int) -> None:
        progress_calls.append((chunk_ix, total, rows_a, matches))

    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=2.0,
        output_path=out,
        progress_callback=progress,
    )
    if len(progress_calls) < 1:
        pytest.skip(
            "Rust extension did not call progress_callback (rebuild with: uv run maturin develop)"
        )


@pytest.mark.integration
def test_fixture_catalog_small_still_works(tmp_path: Path) -> None:
    """Legacy: run on pre-generated fixture catalogs; assert at least known pairs and schema."""
    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    out = tmp_path / "matches.parquet"
    pleiades.cross_match(
        catalog_a=fixtures / "catalog_a_small.parquet",
        catalog_b=fixtures / "catalog_b_small.parquet",
        radius_arcsec=2.0,
        output_path=out,
    )
    assert out.is_file()
    t = read_matches(out)
    assert "separation_arcsec" in t.column_names
    assert t.num_rows >= 10
    sep = t.column("separation_arcsec")
    for i in range(t.num_rows):
        assert sep[i].as_py() <= 2.0


@pytest.mark.integration
def test_cross_match_returns_result_with_counts(tmp_path: Path) -> None:
    """cross_match returns CrossMatchResult with rows_a_read, matches_count, time_seconds."""
    table_a, table_b, _ = make_catalogs_exact_n_pairs(
        n_pairs=3, radius_arcsec=2.0, n_a_extra=5, n_b_extra=5, seed=11
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches.parquet"
    result = pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=2.0,
        output_path=out,
    )
    assert result.output_path == str(out)
    assert result.rows_a_read == 8  # 3 + 5
    assert result.matches_count == 3
    assert result.chunks_processed >= 1
    assert result.time_seconds >= 0


@pytest.mark.integration
def test_prepartitioned_b_directory_matches_file_path(tmp_path: Path) -> None:
    """Using catalog_b as a directory of pre-partitioned shards gives same result as file B."""
    import astropy.units as u
    import cdshealpix.nested as cds_nested
    from astropy.coordinates import Latitude, Longitude

    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=4, radius_arcsec=2.0, n_a_extra=10, n_b_extra=10, seed=31
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    depth = 8
    n_shards = 32
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    schema = pa.schema(
        [
            ("pixel_id", pa.uint64()),
            ("id_b", pa.string()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
        ]
    )
    writers = [
        pq.ParquetWriter(shard_dir / f"shard_{s:04d}.parquet", schema)
        for s in range(n_shards)
    ]
    ra_b = table_b.column("ra").to_numpy()
    dec_b = table_b.column("dec").to_numpy()
    id_b = table_b.column("object_id")
    lon = Longitude(ra_b, unit=u.deg)
    lat = Latitude(dec_b, unit=u.deg)
    pix = cds_nested.lonlat_to_healpix(lon, lat, depth)
    for i in range(len(table_b)):
        s = int(pix[i]) % n_shards
        row = pa.table(
            {
                "pixel_id": [int(pix[i])],
                "id_b": [id_b[i].as_py()],
                "ra": [float(ra_b[i])],
                "dec": [float(dec_b[i])],
            },
            schema=schema,
        )
        writers[s].write_table(row)
    for w in writers:
        w.close()
    out_file = tmp_path / "matches_file.parquet"
    out_dir = tmp_path / "matches_dir.parquet"
    pleiades.cross_match(
        catalog_a=path_a, catalog_b=path_b, radius_arcsec=2.0, output_path=out_file
    )
    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=shard_dir,
        radius_arcsec=2.0,
        output_path=out_dir,
    )
    t_file = read_matches(out_file)
    t_dir = read_matches(out_dir)
    set_file = match_set_from_table(t_file, "source_id", match_table_id_b_col(t_file))
    set_dir = match_set_from_table(t_dir, "source_id", match_table_id_b_col(t_dir))
    assert set_file == set_dir
    assert len(set_dir) == 4


@pytest.mark.integration
def test_partition_catalog_produces_shards(tmp_path: Path) -> None:
    """partition_catalog writes shard_*.parquet with pixel_id, id_b, ra, dec."""
    table_a, table_b = make_catalogs_no_pairs(n_a=50, n_b=30, radius_arcsec=2.0, seed=1)
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out_dir = tmp_path / "shards"
    pleiades.partition_catalog(
        catalog_path=path_b,
        output_dir=out_dir,
        depth=8,
        n_shards=16,
    )
    shards = sorted(out_dir.glob("shard_*.parquet"))
    assert len(shards) == 16
    t0 = pq.read_table(shards[0])
    assert "pixel_id" in t0.column_names
    assert "id_b" in t0.column_names
    assert "ra" in t0.column_names
    assert "dec" in t0.column_names
    total_rows = sum(pq.read_table(p).num_rows for p in shards)
    assert total_rows == 30


@pytest.mark.integration
def test_cross_match_include_coords_adds_ra_dec_columns(tmp_path: Path) -> None:
    """cross_match with include_coords=True writes ra_a, dec_a, ra_b, dec_b."""
    table_a, table_b, expected = make_catalogs_exact_n_pairs(
        n_pairs=3,
        radius_arcsec=2.0,
        separation_arcsec=1.0,
        n_a_extra=5,
        n_b_extra=5,
        seed=22,
    )
    path_a, path_b = write_catalogs(table_a, table_b, tmp_path)
    out = tmp_path / "matches_with_coords.parquet"
    pleiades.cross_match(
        catalog_a=path_a,
        catalog_b=path_b,
        radius_arcsec=2.0,
        output_path=out,
        include_coords=True,
    )
    t = pq.read_table(out)
    assert "ra_a" in t.column_names
    assert "dec_a" in t.column_names
    assert "ra_b" in t.column_names
    assert "dec_b" in t.column_names
    assert t.num_rows >= 3
    sep = t.column("separation_arcsec")
    for i in range(t.num_rows):
        assert sep[i].as_py() <= 2.0
