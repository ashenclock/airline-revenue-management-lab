"""Tests for chunked CSV reading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from airline_rm.data.chunking import read_csv_chunked, count_rows_lazy


@pytest.fixture
def temp_csv(tmp_path) -> Path:
    """Create a temporary CSV file for testing."""
    df = pl.DataFrame({
        "a": list(range(1000)),
        "b": [float(x) for x in range(1000)],
        "c": ["cat"] * 500 + ["dog"] * 500,
    })
    path = tmp_path / "test.csv"
    df.write_csv(path)
    return path


def test_count_rows_lazy(temp_csv):
    """Should count rows without full memory load."""
    count = count_rows_lazy(temp_csv)
    assert count == 1000


def test_chunked_reading(temp_csv):
    """Should yield correct number of chunks."""
    chunks = list(read_csv_chunked(temp_csv, chunk_size=300))
    assert len(chunks) >= 3  # 1000/300 = ~3.3
    total_rows = sum(len(c) for c in chunks)
    assert total_rows == 1000


def test_chunk_columns_preserved(temp_csv):
    """Each chunk should have the same columns."""
    for chunk in read_csv_chunked(temp_csv, chunk_size=500):
        assert "a" in chunk.columns
        assert "b" in chunk.columns
        assert "c" in chunk.columns
