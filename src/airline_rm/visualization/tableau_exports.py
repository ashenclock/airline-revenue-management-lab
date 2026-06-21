"""Tableau export utilities — generate CSVs for Tableau dashboards."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger

from airline_rm.utils.paths import exports_dir, features_dir, models_dir


def export_route_summary(df: pl.DataFrame, output_dir: Path | None = None) -> Path:
    """Export route-level summary for Tableau."""
    if output_dir is None:
        output_dir = exports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    route_summary = (
        df.group_by("ODPairID")
        .agg([
            pl.col("Average_Fare").mean().alias("avg_fare"),
            pl.col("Average_Fare").std().alias("std_fare"),
            pl.col("Average_Fare").median().alias("median_fare"),
            pl.col("Pax").sum().alias("total_pax"),
            pl.col("Carrier").n_unique().alias("n_carriers"),
            pl.col("NonStopMiles").first().alias("distance"),
            pl.col("Market_HHI").first().alias("hhi"),
            pl.len().alias("n_records"),
        ])
        .sort("total_pax", descending=True)
    )

    path = output_dir / "tableau_route_summary.csv"
    route_summary.write_csv(path)
    logger.info(f"Exported route summary: {path} ({len(route_summary)} routes)")
    return path


def export_feature_importance(output_dir: Path | None = None) -> Path:
    """Export feature importance from all models for Tableau."""
    import json

    if output_dir is None:
        output_dir = exports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = models_dir()
    rows = []

    for imp_file in model_dir.glob("*_feature_importance.json"):
        model_name = imp_file.stem.replace("_feature_importance", "")
        with open(imp_file) as f:
            imp = json.load(f)
        for feature, value in imp.items():
            rows.append({"model": model_name, "feature": feature, "importance": value})

    if not rows:
        logger.warning("No feature importance files found")
        return output_dir

    df = pl.DataFrame(rows)
    path = output_dir / "tableau_feature_importance.csv"
    df.write_csv(path)
    logger.info(f"Exported feature importance: {path}")
    return path


def export_revenue_scenarios(
    scenarios_df: pl.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Export revenue scenarios for Tableau."""
    if output_dir is None:
        output_dir = exports_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "tableau_revenue_scenarios.csv"
    scenarios_df.write_csv(path)
    logger.info(f"Exported revenue scenarios: {path}")
    return path


def export_all(df: pl.DataFrame) -> list[Path]:
    """Run all Tableau exports."""
    paths = []
    paths.append(export_route_summary(df))
    paths.append(export_feature_importance())
    logger.info(f"All Tableau exports complete: {len(paths)} files")
    return paths
