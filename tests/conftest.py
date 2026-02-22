"""Pytest configuration and shared fixtures for Pleiades tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests.")
    config.addinivalue_line(
        "markers", "integration: Integration tests (I/O, larger data)."
    )
