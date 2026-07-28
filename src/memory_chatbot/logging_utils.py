"""Logging setup shared by the app, the CLI and the tests."""

from __future__ import annotations

import logging
import os
import sys

PACKAGE_LOGGER = "memory_chatbot"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Attach one stderr handler to the package logger (idempotent).

    Streamlit re-executes the script on every interaction, so this must never
    stack handlers.

    An explicit *level* wins and sticks. A no-argument call only applies the
    ``LOG_LEVEL`` default the first time, because module-level ``get_logger``
    calls run on every lazy import - and those must not undo a level the CLI or
    the caller has already chosen.
    """
    global _configured
    logger = logging.getLogger(PACKAGE_LOGGER)

    if level is not None or not _configured:
        resolved = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(resolved if isinstance(resolved, int) else str(resolved).upper())

    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a package logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
