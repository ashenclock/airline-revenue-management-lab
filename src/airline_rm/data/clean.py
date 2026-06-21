"""Data cleaning and validation pipeline."""

from __future__ import annotations

import polars as pl
from loguru import logger

from airline_rm.data.schema import (
    TARGET_COLUMN,
    DataValidationResult,
    FREQ_ENCODED_COLUMNS,
)
from airline_rm.utils.memory import log_memory


def validate_dataframe(df: pl.DataFrame, target_col: str = TARGET_COLUMN) -> DataValidationResult:
    """Run validation checks on the dataframe.

    Checks:
    - No negative fares
    - No zero fares
    - No negative distances
    - No impossible circuity values
    - Duplicate rows
    - Missing target values
    """
    result = DataValidationResult(total_rows=len(df))

    # Negative fares
    if target_col in df.columns:
        result.negative_fares = df.filter(pl.col(target_col) < 0).height
        if result.negative_fares > 0:
            result.issues.append(
                f"{result.negative_fares} rows with negative {target_col}"
            )

        result.zero_fares = df.filter(pl.col(target_col) == 0).height
        if result.zero_fares > 0:
            result.issues.append(
                f"{result.zero_fares} rows with zero {target_col}"
            )

        result.missing_target = df[target_col].null_count()
        if result.missing_target > 0:
            result.issues.append(
                f"{result.missing_target} rows with missing {target_col}"
            )

    # Negative distances
    for col in ["NonStopMiles", "MktMilesFlown"]:
        if col in df.columns:
            neg = df.filter(pl.col(col) < 0).height
            if neg > 0:
                result.negative_distances += neg
                result.issues.append(f"{neg} rows with negative {col}")

    # Impossible circuity (should be >= 1.0)
    if "Circuity" in df.columns:
        result.impossible_circuity = df.filter(pl.col("Circuity") < 0.5).height
        if result.impossible_circuity > 0:
            result.issues.append(
                f"{result.impossible_circuity} rows with impossibly low Circuity"
            )

    # Duplicates
    result.duplicate_rows = len(df) - df.unique().height
    if result.duplicate_rows > 0:
        result.issues.append(f"{result.duplicate_rows} duplicate rows")

    return result


def clean_dataframe(
    df: pl.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_freq_encoded: bool = True,
    drop_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Clean the dataframe by removing invalid rows and columns.

    Steps:
    1. Drop pre-computed frequency encoded columns (potential leakage)
    2. Drop additional specified columns
    3. Remove rows with negative/zero target
    4. Remove rows with missing target
    5. Remove rows with negative distances
    6. Remove exact duplicates
    7. Downcast numeric types for memory efficiency
    """
    log_memory("clean — start")
    initial_rows = len(df)

    # 1. Drop frequency-encoded columns
    cols_to_drop = []
    if drop_freq_encoded:
        cols_to_drop.extend([c for c in FREQ_ENCODED_COLUMNS if c in df.columns])
    if drop_columns:
        cols_to_drop.extend([c for c in drop_columns if c in df.columns])

    if cols_to_drop:
        df = df.drop(cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")

    # 2. Remove invalid target values
    if target_col in df.columns:
        df = df.filter(
            pl.col(target_col).is_not_null() & (pl.col(target_col) > 0)
        )

    # 3. Remove negative distances
    for col in ["NonStopMiles", "MktMilesFlown"]:
        if col in df.columns:
            df = df.filter(pl.col(col) >= 0)

    # 4. Remove duplicates
    df = df.unique()

    # 5. Downcast numeric types
    df = _downcast_numerics(df)

    removed = initial_rows - len(df)
    logger.info(f"Cleaning: {initial_rows:,} → {len(df):,} rows (removed {removed:,})")
    log_memory("clean — end")

    return df


def _downcast_numerics(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast Float64 → Float32 and Int64 → Int32 where safe."""
    casts = {}
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Float64:
            casts[col_name] = pl.Float32
        elif dtype == pl.Int64:
            col_max = df[col_name].max()
            col_min = df[col_name].min()
            if col_min is not None and col_max is not None:
                if col_min >= -(2**31) and col_max < 2**31:
                    casts[col_name] = pl.Int32

    if casts:
        df = df.cast(casts)
        logger.debug(f"Downcasted {len(casts)} columns for memory efficiency")

    return df
