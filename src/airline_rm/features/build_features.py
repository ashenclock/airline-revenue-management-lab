"""Feature building orchestrator — loads processed data, builds all features."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from airline_rm.config.settings import Settings, load_settings
from airline_rm.data.schema import (
    CATEGORICAL_COLUMNS,
    ID_COLUMNS,
    TARGET_COLUMN,
)
from airline_rm.features.categoricals import frequency_encode, target_encode_cv
from airline_rm.features.time_features import (
    build_carrier_features,
    build_market_features,
    build_route_features,
)
from airline_rm.utils.memory import MemoryGuard, log_memory
from airline_rm.utils.paths import features_dir, processed_dir


def build_features(settings: Settings | None = None) -> Path:
    """Build the full feature dataset from processed Parquet.

    Steps:
    1. Load processed Parquet
    2. Add route-level features
    3. Add carrier-level features
    4. Add market-level features
    5. Frequency encode ID columns
    6. Target encode categorical columns (with CV isolation)
    7. Save to data/features/features.parquet

    Returns:
        Path to the features Parquet file.
    """
    if settings is None:
        settings = load_settings()

    input_path = processed_dir() / "airline_fares.parquet"
    output_path = features_dir() / "features.parquet"

    if output_path.exists():
        logger.info(f"Features already exist: {output_path}. Skipping.")
        return output_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {input_path}. "
            "Run 'airline-rm build-parquet' first."
        )

    with MemoryGuard(settings.memory.limit_gb, "Feature building"):
        logger.info(f"Loading processed data from {input_path}")
        df = pl.read_parquet(input_path)
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        # Route-level features
        df = build_route_features(df)
        log_memory("after route features")

        # Carrier-level features
        df = build_carrier_features(df)
        log_memory("after carrier features")

        # Market-level features
        df = build_market_features(df)
        log_memory("after market features")

        # Frequency encoding for ID columns
        id_cols_present = [c for c in ID_COLUMNS if c in df.columns]
        if id_cols_present:
            df = frequency_encode(df, id_cols_present)
            log_memory("after frequency encoding")

        # Target encoding for categorical columns (with CV fold isolation)
        target = settings.dataset.target_column
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        for col in cat_cols:
            df = target_encode_cv(
                df,
                column=col,
                target=target,
                n_folds=5,
                seed=settings.splitting.random_seed,
            )
        log_memory("after target encoding")

        # Save
        df.write_parquet(output_path, compression="snappy")
        size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"Features saved: {output_path} ({size_mb:.1f} MB)")
        logger.info(f"Final shape: {len(df):,} rows × {len(df.columns)} columns")
        logger.info(f"Feature columns: {df.columns}")

    return output_path
