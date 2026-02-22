"""Unit tests for the astrojoin CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
class TestCLI:
    """Tests for the astrojoin command-line interface."""

    def test_astrojoin_help(self) -> None:
        """astrojoin --help exits 0 and prints usage."""
        result = subprocess.run(
            [sys.executable, "-m", "astrojoin.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "cross-match" in result.stdout or "usage" in result.stdout.lower()

    def test_cross_match_help(self) -> None:
        """astrojoin cross-match --help shows required args."""
        result = subprocess.run(
            [sys.executable, "-m", "astrojoin.cli", "cross-match", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "catalog_a" in result.stdout or "radius" in result.stdout

    def test_summarize_matches_help(self) -> None:
        """astrojoin summarize-matches --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "astrojoin.cli", "summarize-matches", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_cone_search_help(self) -> None:
        """astrojoin cone-search --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "astrojoin.cli", "cone-search", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_partition_catalog_help(self) -> None:
        """astrojoin partition-catalog --help runs."""
        result = subprocess.run(
            [sys.executable, "-m", "astrojoin.cli", "partition-catalog", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

    def test_cross_match_run_with_fixtures(self, tmp_path: Path) -> None:
        """astrojoin cross-match runs and writes output (integration-style)."""
        fixtures = Path(__file__).resolve().parent.parent / "fixtures"
        out = tmp_path / "matches.parquet"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "astrojoin.cli",
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
