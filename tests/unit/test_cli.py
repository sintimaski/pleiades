"""Unit tests for the pleiades CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
class TestCLI:
    """Tests for the pleiades command-line interface."""

    def test_pleiades_help(self) -> None:
        """pleiades --help exits 0 and prints usage."""
        result = subprocess.run(
            [sys.executable, "-m", "pleiades.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "cross-match" in result.stdout or "usage" in result.stdout.lower()

    def test_cross_match_help(self) -> None:
        """pleiades cross-match --help shows required args."""
        result = subprocess.run(
            [sys.executable, "-m", "pleiades.cli", "cross-match", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "catalog_a" in result.stdout or "radius" in result.stdout

    def test_summarize_matches_help(self) -> None:
        """pleiades summarize-matches --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "pleiades.cli", "summarize-matches", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_cone_search_help(self) -> None:
        """pleiades cone-search --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "pleiades.cli", "cone-search", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_partition_catalog_help(self) -> None:
        """pleiades partition-catalog --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "pleiades.cli", "partition-catalog", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_cross_match_run_with_fixtures(self, tmp_path: Path) -> None:
        """pleiades cross-match runs and writes output (integration-style)."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "matches.parquet"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pleiades.cli",
                "cross-match",
                str(fixtures / "catalog_a_small.parquet"),
                str(fixtures / "catalog_b_small.parquet"),
                "-r",
                "2.0",
                "-o",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert out.is_file()
        assert "matches" in result.stdout.lower() or "Wrote" in result.stdout

    def test_cli_exit_1_and_stderr_on_missing_catalog(self, tmp_path: Path) -> None:
        """CLI exits with 1 and prints to stderr when a catalog file is missing."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pleiades.cli",
                "cross-match",
                "/nonexistent/catalog_a.parquet",
                str(fixtures / "catalog_b_small.parquet"),
                "-r",
                "2.0",
                "-o",
                str(tmp_path / "out.parquet"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "pleiades:" in result.stderr
        assert "not found" in result.stderr.lower() or "nonexistent" in result.stderr

    def test_cli_exit_1_and_stderr_on_invalid_radius(self, tmp_path: Path) -> None:
        """CLI exits with 1 and prints to stderr when radius is invalid (e.g. zero)."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pleiades.cli",
                "cross-match",
                str(fixtures / "catalog_a_small.parquet"),
                str(fixtures / "catalog_b_small.parquet"),
                "-r",
                "0",
                "-o",
                str(tmp_path / "out.parquet"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "pleiades:" in result.stderr
        assert "positive" in result.stderr.lower() or "radius" in result.stderr.lower()
