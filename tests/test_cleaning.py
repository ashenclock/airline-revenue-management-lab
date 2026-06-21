"""Tests for data cleaning and validation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from airline_rm.data.clean import clean_dataframe, validate_dataframe


def test_validate_clean_data(sample_df):
    """Clean data should pass validation."""
    result = validate_dataframe(sample_df)
    assert result.total_rows == len(sample_df)
    assert result.negative_fares == 0  # Our fixture has positive fares


def test_validate_negative_fares():
    """Should detect negative fares."""
    df = pl.DataFrame({
        "Average_Fare": [-10.0, 100.0, 200.0],
        "NonStopMiles": [500.0, 800.0, 1200.0],
        "MktMilesFlown": [500.0, 800.0, 1200.0],
    })
    result = validate_dataframe(df)
    assert result.negative_fares == 1
    assert not result.is_clean


def test_validate_negative_distances():
    """Should detect negative distances."""
    df = pl.DataFrame({
        "Average_Fare": [100.0, 200.0],
        "NonStopMiles": [-500.0, 800.0],
        "MktMilesFlown": [500.0, 800.0],
    })
    result = validate_dataframe(df)
    assert result.negative_distances == 1


def test_clean_removes_negative_fares():
    """Cleaning should remove rows with negative fares."""
    df = pl.DataFrame({
        "Average_Fare": [-10.0, 0.0, 100.0, 200.0],
        "NonStopMiles": [500.0, 800.0, 1000.0, 1200.0],
        "MktMilesFlown": [500.0, 800.0, 1000.0, 1200.0],
    })
    cleaned = clean_dataframe(df, drop_freq_encoded=False)
    assert len(cleaned) == 2  # Only positive fares kept


def test_clean_drops_freq_encoded():
    """Should drop pre-computed frequency encoding columns."""
    df = pl.DataFrame({
        "Average_Fare": [100.0, 200.0],
        "Carrier_freq": [0.5, 0.3],
        "ODPairID_freq": [0.1, 0.2],
    })
    cleaned = clean_dataframe(df, drop_freq_encoded=True)
    assert "Carrier_freq" not in cleaned.columns
    assert "ODPairID_freq" not in cleaned.columns


def test_clean_removes_duplicates():
    """Should remove exact duplicate rows."""
    df = pl.DataFrame({
        "Average_Fare": [100.0, 100.0, 200.0],
        "NonStopMiles": [500.0, 500.0, 800.0],
    })
    cleaned = clean_dataframe(df, drop_freq_encoded=False)
    assert len(cleaned) == 2
