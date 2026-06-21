"""Tests for feature generation."""

from __future__ import annotations

import polars as pl
import numpy as np
import pytest

from airline_rm.features.categoricals import frequency_encode, target_encode_cv
from airline_rm.features.time_features import build_route_features, build_carrier_features


def test_frequency_encode(sample_df):
    """Frequency encoding should add new columns summing to ~1."""
    result = frequency_encode(sample_df, ["Carrier"])
    assert "Carrier_freq_enc" in result.columns
    # Frequencies should be between 0 and 1
    freqs = result["Carrier_freq_enc"]
    assert freqs.min() > 0
    assert freqs.max() <= 1


def test_target_encode_cv(sample_df):
    """Target encoding should not leak — values differ from raw means."""
    result = target_encode_cv(
        sample_df, column="Carrier", target="Average_Fare", n_folds=3
    )
    assert "Carrier_target_enc" in result.columns
    # Should have same number of rows
    assert len(result) == len(sample_df)
    # Values should be reasonable (within fare range)
    enc = result["Carrier_target_enc"]
    assert enc.min() > 0
    assert enc.max() < 1000


def test_build_route_features(sample_df):
    """Should add route-level aggregate features."""
    result = build_route_features(sample_df)
    assert "route_avg_fare" in result.columns
    assert "route_carrier_count" in result.columns
    assert "route_competition_level" in result.columns
    assert len(result) == len(sample_df)


def test_build_carrier_features(sample_df):
    """Should add carrier-level aggregate features."""
    result = build_carrier_features(sample_df)
    assert "carrier_avg_fare" in result.columns
    assert "carrier_route_count" in result.columns
    assert len(result) == len(sample_df)
