"""Structured logging setup using loguru."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with structured output.

    Removes default handler and adds a clean stderr handler with the specified level.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def get_logger(name: str = "airline_rm") -> "logger":
    """Return a contextualized logger instance."""
    return logger.bind(module=name)
