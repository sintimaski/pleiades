"""Command-line interface for AstroJoin."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _path(s: str) -> Path:
    """Convert a CLI path string to a Path (used for type consistency)."""
    return Path(s)


def cmd_cross_match(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run cross-match from CLI."""
    import astrojoin

    result = astrojoin.cross_match(
        catalog_a=args.catalog_a,
        catalog_b=args.catalog_b,
        radius_arcsec=float(args.radius),
        output_path=args.output,
        ra_col=args.ra_col or "ra",
        dec_col=args.dec_col or "dec",
        id_col_a=args.id_col_a,
        id_col_b=args.id_col_b,
        n_nearest=int(args.n_nearest) if args.n_nearest else None,
        use_rust=args.rust,
    )
    print(
        f"Wrote {result.matches_count} matches to {result.output_path} "
        f"({result.time_seconds:.2f}s)"
    )
    return 0


def cmd_summarize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Summarize a matches file."""
    import astrojoin

    summary = astrojoin.summarize_matches(args.matches)
    print(f"Matches: {summary.num_matches}")
    print(f"Unique id_a: {summary.num_unique_id_a}, id_b: {summary.num_unique_id_b}")
    print(
        f"Separation (arcsec): min={summary.separation_arcsec_min:.4f} "
        f"max={summary.separation_arcsec_max:.4f} mean={summary.separation_arcsec_mean:.4f}"
    )
    if summary.separation_arcsec_median is not None:
        print(f"  median={summary.separation_arcsec_median:.4f}")
    return 0


def cmd_cone_search(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run cone search from CLI."""
    import astrojoin

    n = astrojoin.cone_search(
        catalog_path=args.catalog,
        ra_deg=float(args.ra),
        dec_deg=float(args.dec),
        radius_arcsec=float(args.radius),
        output_path=args.output,
    )
    print(f"Wrote {n} rows to {args.output}")
    return 0


def cmd_partition(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Partition a catalog by HEALPix into shards."""
    import astrojoin

    astrojoin.partition_catalog(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        depth=args.depth,
        n_shards=args.n_shards,
        ra_col=args.ra_col or "ra",
        dec_col=args.dec_col or "dec",
        id_col=args.id_col,
    )
    print(f"Partitioned {args.catalog} -> {args.output_dir}")
    return 0


def main() -> int:
    """Entry point for astrojoin command."""
    parser = argparse.ArgumentParser(
        prog="astrojoin",
        description="Out-of-core spatial cross-matcher for astronomical catalogs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cross-match
    p_cm = subparsers.add_parser("cross-match", help="Cross-match two Parquet catalogs")
    p_cm.add_argument("catalog_a", type=_path, help="Path to catalog A (Parquet)")
    p_cm.add_argument("catalog_b", type=_path, help="Path to catalog B or shard directory")
    p_cm.add_argument("-r", "--radius", required=True, help="Match radius (arcsec)")
    p_cm.add_argument("-o", "--output", required=True, type=_path, help="Output matches Parquet")
    p_cm.add_argument("--ra-col", default=None, help="RA column name (default: ra)")
    p_cm.add_argument("--dec-col", default=None, help="Dec column name (default: dec)")
    p_cm.add_argument("--id-col-a", default=None, help="ID column in catalog A")
    p_cm.add_argument("--id-col-b", default=None, help="ID column in catalog B")
    p_cm.add_argument("--n-nearest", default=None, help="Keep only n best matches per id_a (e.g. 1)")
    p_cm.add_argument(
        "--rust",
        action="store_true",
        default=True,
        help="Use Rust engine (default)",
    )
    p_cm.add_argument(
        "--no-rust",
        action="store_false",
        dest="rust",
        help="Use Python implementation instead (slow)",
    )
    p_cm.set_defaults(func=cmd_cross_match)

    # summarize-matches
    p_sm = subparsers.add_parser("summarize-matches", help="Summarize a matches Parquet file")
    p_sm.add_argument("matches", type=_path, help="Path to matches Parquet")
    p_sm.set_defaults(func=cmd_summarize)

    # cone-search
    p_cs = subparsers.add_parser("cone-search", help="Cone search: rows within radius of a point")
    p_cs.add_argument("catalog", type=_path, help="Path to catalog Parquet")
    p_cs.add_argument("ra", help="Center RA (degrees)")
    p_cs.add_argument("dec", help="Center Dec (degrees)")
    p_cs.add_argument("-r", "--radius", required=True, help="Radius (arcsec)")
    p_cs.add_argument("-o", "--output", required=True, type=_path, help="Output Parquet path")
    p_cs.set_defaults(func=cmd_cone_search)

    # partition-catalog
    p_pc = subparsers.add_parser(
        "partition-catalog",
        help="Partition catalog by HEALPix into shard Parquet files",
    )
    p_pc.add_argument("catalog", type=_path, help="Path to catalog Parquet")
    p_pc.add_argument("output_dir", type=_path, help="Output directory for shards")
    p_pc.add_argument("--depth", type=int, default=8, help="HEALPix depth (default: 8)")
    p_pc.add_argument("--n-shards", type=int, default=512, help="Number of shards (default: 512)")
    p_pc.add_argument("--ra-col", default=None, help="RA column name")
    p_pc.add_argument("--dec-col", default=None, help="Dec column name")
    p_pc.add_argument("--id-col", default=None, help="ID column name")
    p_pc.set_defaults(func=cmd_partition)

    args = parser.parse_args()
    return args.func(parser, args)


if __name__ == "__main__":
    sys.exit(main())
