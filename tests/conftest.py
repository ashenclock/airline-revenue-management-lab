"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a small synthetic airline fare dataset for testing."""
    np.random.seed(42)
    n = 200

    return pl.DataFrame({
        "MktCoupons": np.random.randint(1, 4, n),
        "OriginCityMarketID": np.random.randint(30000, 30010, n),
        "DestCityMarketID": np.random.randint(30000, 30010, n),
        "OriginAirportID": np.random.randint(10000, 10005, n),
        "DestAirportID": np.random.randint(10000, 10005, n),
        "Carrier": np.random.choice(["AA", "UA", "DL", "WN", "B6"], n),
        "NonStopMiles": np.random.uniform(100, 3000, n).astype(np.float32),
        "RoundTrip": np.random.randint(0, 2, n),
        "ODPairID": np.random.randint(1, 20, n),
        "Pax": np.random.randint(100, 50000, n),
        "CarrierPax": np.random.randint(50, 25000, n),
        "Average_Fare": np.random.uniform(50, 800, n).astype(np.float32),
        "Market_share": np.random.uniform(0, 1, n).astype(np.float32),
        "Market_HHI": np.random.uniform(1000, 10000, n).astype(np.float32),
        "LCC_Comp": np.random.randint(0, 2, n),
        "Multi_Airport": np.random.randint(0, 2, n),
        "Circuity": np.random.uniform(1.0, 2.0, n).astype(np.float32),
        "Slot": np.random.randint(0, 2, n),
        "Non_Stop": np.random.randint(0, 2, n),
        "MktMilesFlown": np.random.uniform(100, 5000, n).astype(np.float32),
    })


@pytest.fixture
def feature_cols() -> list[str]:
    """Feature columns for testing."""
    return [
        "MktCoupons", "NonStopMiles", "RoundTrip", "Pax", "CarrierPax",
        "Market_share", "Market_HHI", "LCC_Comp", "Multi_Airport",
        "Circuity", "Slot", "Non_Stop", "MktMilesFlown",
    ]
