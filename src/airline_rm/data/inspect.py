"""Inspect raw dataset files without loading fully into memory."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from loguru import logger

from airline_rm.utils.memory import log_memory
from airline_rm.utils.paths import raw_dir, reports_dir


def inspect_dataset(data_path: Path | None = None) -> dict:
    """Inspect raw CSV files using Polars lazy scanning.

    Prints and returns a summary including:
    - file names, sizes
    - column names, dtypes
    - null counts (sampled)
    - row counts
    - basic statistics
    """
    if data_path is None:
        data_path = raw_dir()

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {data_path}")
        logger.info("Run 'airline-rm download' first.")
        return {}

    report = {}

    for csv_path in csv_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Inspecting: {csv_path.name}")
        logger.info(f"  Size: {csv_path.stat().st_size / (1024**2):.1f} MB")

        log_memory("pre-inspect")

        # Use Polars lazy scan for memory efficiency
        lf = pl.scan_csv(csv_path)
        schema = lf.collect_schema()

        # Collect row count efficiently
        row_count = lf.select(pl.len()).collect().item()

        logger.info(f"  Rows: {row_count:,}")
        logger.info(f"  Columns: {len(schema)}")

        # Column details
        col_info = {}
        for col_name, dtype in schema.items():
            col_info[col_name] = str(dtype)
            logger.info(f"    {col_name}: {dtype}")

        # Sample first 1000 rows for null analysis
        sample = lf.head(1000).collect()
        null_counts = {}
        for col in sample.columns:
            nc = sample[col].null_count()
            if nc > 0:
                null_counts[col] = nc

        if null_counts:
            logger.info("  Nulls (first 1000 rows):")
            for col, count in null_counts.items():
                logger.info(f"    {col}: {count}")
        else:
            logger.info("  No nulls in first 1000 rows")

        # Full null count (lazy)
        full_null = (
            lf.select([pl.col(c).null_count().alias(c) for c in schema.names()])
            .collect()
            .to_dicts()[0]
        )
        full_null = {k: v for k, v in full_null.items() if v > 0}

        # Basic stats on numeric columns
        numeric_cols = [
            name for name, dtype in schema.items()
            if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32)
        ]
        stats = {}
        if numeric_cols:
            stats_df = lf.select(numeric_cols).describe()
            stats = stats_df.to_pandas().set_index("statistic").to_dict()

        file_report = {
            "file_name": csv_path.name,
            "size_mb": round(csv_path.stat().st_size / (1024**2), 2),
            "row_count": row_count,
            "column_count": len(schema),
            "columns": col_info,
            "null_counts_full": full_null,
            "numeric_stats": stats,
        }
        report[csv_path.name] = file_report

        log_memory("post-inspect")

    # Save report
    report_path = reports_dir() / "inspection_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nInspection report saved to {report_path}")

    return report
