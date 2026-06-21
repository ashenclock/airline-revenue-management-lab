"""Data schema definitions and validation rules."""

from __future__ import annotations

from pydantic import BaseModel, Field


# Expected columns from the airline market fare prediction dataset
EXPECTED_COLUMNS: list[str] = [
    "MktCoupons",
    "OriginCityMarketID",
    "DestCityMarketID",
    "OriginAirportID",
    "DestAirportID",
    "Carrier",
    "NonStopMiles",
    "RoundTrip",
    "ODPairID",
    "Pax",
    "CarrierPax",
    "Average_Fare",
    "Market_share",
    "Market_HHI",
    "LCC_Comp",
    "Multi_Airport",
    "Circuity",
    "Slot",
    "Non_Stop",
    "MktMilesFlown",
    # Pre-computed frequency encodings (candidates for dropping)
    "OriginCityMarketID_freq",
    "DestCityMarketID_freq",
    "OriginAirportID_freq",
    "DestAirportID_freq",
    "Carrier_freq",
    "ODPairID_freq",
]

# Columns that are frequency encodings (potential leakage)
FREQ_ENCODED_COLUMNS: list[str] = [c for c in EXPECTED_COLUMNS if c.endswith("_freq")]

# Categorical columns
CATEGORICAL_COLUMNS: list[str] = ["Carrier"]

# ID / high-cardinality columns (encode, don't one-hot)
ID_COLUMNS: list[str] = [
    "OriginCityMarketID",
    "DestCityMarketID",
    "OriginAirportID",
    "DestAirportID",
    "ODPairID",
]

# Binary indicator columns
BINARY_COLUMNS: list[str] = [
    "RoundTrip",
    "LCC_Comp",
    "Multi_Airport",
    "Slot",
    "Non_Stop",
]

# Core numeric feature columns
NUMERIC_FEATURE_COLUMNS: list[str] = [
    "MktCoupons",
    "NonStopMiles",
    "Pax",
    "CarrierPax",
    "Market_share",
    "Market_HHI",
    "Circuity",
    "MktMilesFlown",
]

TARGET_COLUMN: str = "Average_Fare"

# Demand proxy candidates
DEMAND_PROXY_COLUMNS: list[str] = ["Pax", "CarrierPax"]


class DataValidationResult(BaseModel):
    """Result of data validation checks."""

    total_rows: int = 0
    negative_fares: int = 0
    zero_fares: int = 0
    negative_distances: int = 0
    impossible_circuity: int = 0
    duplicate_rows: int = 0
    missing_target: int = 0
    issues: list[str] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0
