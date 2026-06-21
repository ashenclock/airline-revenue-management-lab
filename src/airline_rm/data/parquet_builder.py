"""Convert raw CSVs to compressed Parquet files."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from airline_rm.data.clean import clean_dataframe
from airline_rm.config.settings import Settings, load_settings
from airline_rm.utils.memory import MemoryGuard, log_memory
from airline_rm.utils.paths import processed_dir, raw_dir


def build_parquet(settings: Settings | None = None) -> Path:
    """Convert raw CSV to cleaned, compressed Parquet.

    Steps:
    1. Scan raw CSV with Polars lazy mode
    2. Collect (dataset fits in memory at ~200MB)
    3. Clean and validate
    4. Write compressed Parquet to data/processed/

    Returns:
        Path to the output Parquet file.
    """
    if settings is None:
        settings = load_settings()

    raw = raw_dir()
    output = processed_dir()

    csv_files = sorted(raw.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files in {raw}. Run 'airline-rm download' first.")
        raise FileNotFoundError(f"No CSV files in {raw}")

    output_path = output / "airline_fares.parquet"

    if output_path.exists():
        logger.info(f"Parquet already exists: {output_path}. Skipping.")
        return output_path

    with MemoryGuard(settings.memory.limit_gb, "Parquet build"):
        # For this dataset (~200MB CSV, ~1.5M rows), Polars can handle it
        dfs = []
        for csv_path in csv_files:
            logger.info(f"Reading {csv_path.name}...")
            df = pl.scan_csv(csv_path).collect()
            logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
            dfs.append(df)

        if len(dfs) > 1:
            df = pl.concat(dfs)
            logger.info(f"Concatenated: {len(df):,} total rows")
        else:
            df = dfs[0]

        # Clean
        df = clean_dataframe(
            df,
            target_col=settings.dataset.target_column,
            drop_freq_encoded=True,
            drop_columns=settings.dataset.drop_columns,
        )

        # Write Parquet
        df.write_parquet(output_path, compression="snappy")
        size_mb = output_path.stat().st_size / (1024**2)
        logger.info(f"Parquet written: {output_path} ({size_mb:.1f} MB)")
        log_memory("post-parquet")

    return output_path
