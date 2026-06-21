"""Memory monitoring and guards for M1 8GB constraint."""

from __future__ import annotations

import gc
from typing import Optional

import psutil
from loguru import logger


def get_memory_usage_gb() -> float:
    """Return current process RSS memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def get_available_memory_gb() -> float:
    """Return available system memory in GB."""
    return psutil.virtual_memory().available / (1024**3)


def log_memory(context: str = "") -> float:
    """Log current memory usage and return RSS in GB."""
    used = get_memory_usage_gb()
    available = get_available_memory_gb()
    msg = f"Memory — Process: {used:.2f} GB | Available: {available:.2f} GB"
    if context:
        msg = f"[{context}] {msg}"
    logger.info(msg)
    return used


def check_memory_limit(limit_gb: float = 5.0, context: str = "") -> bool:
    """Check if current memory usage is below the limit.

    Returns True if safe, raises MemoryError if exceeded.
    """
    used = get_memory_usage_gb()
    if used > limit_gb:
        msg = (
            f"Memory limit exceeded: {used:.2f} GB used > {limit_gb:.1f} GB limit. "
            f"Context: {context}"
        )
        logger.error(msg)
        raise MemoryError(msg)
    return True


def force_gc(context: str = "") -> None:
    """Force garbage collection and log memory change."""
    before = get_memory_usage_gb()
    gc.collect()
    after = get_memory_usage_gb()
    freed = before - after
    if freed > 0.01:
        logger.debug(f"GC freed {freed:.3f} GB {f'({context})' if context else ''}")


class MemoryGuard:
    """Context manager that checks memory before and after a block.

    Usage:
        with MemoryGuard(limit_gb=5.0, context="XGBoost training"):
            model.fit(X, y)
    """

    def __init__(self, limit_gb: float = 5.0, context: str = ""):
        self.limit_gb = limit_gb
        self.context = context
        self._start_usage: Optional[float] = None

    def __enter__(self) -> "MemoryGuard":
        self._start_usage = log_memory(f"{self.context} — START")
        check_memory_limit(self.limit_gb, self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        end_usage = log_memory(f"{self.context} — END")
        if self._start_usage is not None:
            delta = end_usage - self._start_usage
            logger.info(
                f"[{self.context}] Memory delta: {delta:+.3f} GB"
            )
        force_gc(self.context)
        return None
