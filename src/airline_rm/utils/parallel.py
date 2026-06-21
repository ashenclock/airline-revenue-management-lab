"""Safe parallelism utilities for Mac M1 with limited RAM."""

from __future__ import annotations

import os

import psutil


def safe_n_jobs(requested: int = 2, memory_per_job_gb: float = 0.5) -> int:
    """Return a safe n_jobs value based on available memory.

    On Mac M1 with 8GB, we default to conservative parallelism to avoid OOM.

    Args:
        requested: Desired number of jobs.
        memory_per_job_gb: Estimated memory per parallel job.

    Returns:
        Clamped n_jobs value.
    """
    available_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = os.cpu_count() or 2

    # Max jobs that fit in memory
    max_by_memory = max(1, int(available_gb / memory_per_job_gb))
    # Don't exceed CPU count
    max_by_cpu = cpu_count

    # Resolve -1 (all CPUs)
    if requested == -1:
        requested = cpu_count

    safe = min(requested, max_by_memory, max_by_cpu)
    return max(1, safe)
