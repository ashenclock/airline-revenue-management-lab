"""Chunked CSV reading for memory-constrained environments."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import polars as pl
from loguru import logger

from airline_rm.utils.memory import check_memory_limit, log_memory


def read_csv_chunked(
    path: Path,
    chunk_size: int = 100_000,
    memory_limit_gb: float = 5.0,
) -> Generator[pl.DataFrame, None, None]:
    """Read a CSV file in chunks using Polars.

    Yields DataFrames of at most `chunk_size` rows. Checks memory
    before each chunk to fail gracefully.

    Args:
        path: Path to the CSV file.
        chunk_size: Number of rows per chunk.
        memory_limit_gb: Memory limit in GB.

    Yields:
        Polars DataFrame chunks.
    """
    logger.info(f"Reading {path.name} in chunks of {chunk_size:,}")

    reader = pl.read_csv_batched(path, batch_size=chunk_size)
    batch_num = 0

    while True:
        check_memory_limit(memory_limit_gb, f"chunk {batch_num}")
        batches = reader.next_batches(1)
        if batches is None or len(batches) == 0:
            break
        batch_num += 1
        chunk = batches[0]
        logger.debug(f"  Chunk {batch_num}: {len(chunk):,} rows")
        yield chunk

    logger.info(f"Finished reading {batch_num} chunks from {path.name}")


def count_rows_lazy(path: Path) -> int:
    """Count rows in a CSV without loading it fully into memory."""
    return pl.scan_csv(path).select(pl.len()).collect().item()


def read_csv_lazy(path: Path) -> pl.LazyFrame:
    """Return a Polars LazyFrame for deferred computation."""
    return pl.scan_csv(path)
