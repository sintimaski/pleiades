"""Logging configuration for Pleiades.

Log level is controlled by PLEIADES_VERBOSE: set to 1 (or true) for INFO,
so that cross_match and CLI emit progress and timing. Default is WARNING
to avoid noise when used as a library.
"""

from __future__ import annotations

import logging
import os

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Return the Pleiades package logger, configuring it on first use."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger("pleiades")
        if not _LOGGER.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(levelname)s [pleiades] %(message)s")
            )
            _LOGGER.addHandler(handler)
        verbose = os.environ.get("PLEIADES_VERBOSE", "").strip().lower()
        if verbose in ("1", "true", "yes"):
            _LOGGER.setLevel(logging.INFO)
        else:
            _LOGGER.setLevel(logging.WARNING)
    return _LOGGER
