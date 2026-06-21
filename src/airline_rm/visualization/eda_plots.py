"""EDA plots — save publication-quality visualizations to data/reports/eda/."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger

from airline_rm.data.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from airline_rm.utils.memory import log_memory
from airline_rm.utils.paths import reports_dir


def run_eda(df: pl.DataFrame, target_col: str = TARGET_COLUMN) -> Path:
    """Run full EDA and save plots to data/reports/eda/.

    Generates:
    - Target distribution
    - Feature distributions
    - Distance vs fare scatter
    - Carrier boxplots
    - Correlation heatmap
    - Missing values summary
    - Feature cardinality
    - Market HHI distribution
    - Top routes by fare
    """
    eda_dir = reports_dir("eda")
    eda_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    logger.info(f"Running EDA on {len(df):,} rows, saving to {eda_dir}")

    # Convert to pandas for seaborn compatibility (sample if too large)
    if len(df) > 500_000:
        pdf = df.sample(n=500_000, seed=42).to_pandas()
        logger.info("Sampled 500K rows for visualization")
    else:
        pdf = df.to_pandas()

    # 1. Target distribution
    _plot_target_distribution(pdf, target_col, eda_dir)

    # 2. Distance vs Fare
    _plot_distance_vs_fare(pdf, target_col, eda_dir)

    # 3. Carrier fare boxplot
    if "Carrier" in pdf.columns:
        _plot_carrier_fares(pdf, target_col, eda_dir)

    # 4. Correlation heatmap
    _plot_correlation_heatmap(pdf, eda_dir)

    # 5. Missing values
    _plot_missing_values(df, eda_dir)

    # 6. Feature cardinality
    _plot_feature_cardinality(df, eda_dir)

    # 7. Market HHI distribution
    if "Market_HHI" in pdf.columns:
        _plot_hhi_distribution(pdf, eda_dir)

    # 8. Top routes by fare
    _plot_top_routes(df, target_col, eda_dir)

    # 9. Binary feature distributions
    _plot_binary_features(pdf, target_col, eda_dir)

    # 10. Summary stats table
    _save_summary_stats(df, target_col, eda_dir)

    log_memory("post-EDA")
    logger.info(f"EDA complete. {len(list(eda_dir.iterdir()))} files saved.")
    return eda_dir


def _plot_target_distribution(pdf, target_col: str, eda_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pdf[target_col].dropna(), bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].set_title(f"{target_col} Distribution")
    axes[0].set_xlabel("Fare ($)")
    axes[0].set_ylabel("Count")

    # Log-transformed
    log_vals = np.log1p(pdf[target_col].dropna())
    axes[1].hist(log_vals, bins=100, color="#FF9800", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"log(1 + {target_col}) Distribution")
    axes[1].set_xlabel("Log Fare")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(eda_dir / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: target_distribution.png")


def _plot_distance_vs_fare(pdf, target_col: str, eda_dir: Path) -> None:
    if "NonStopMiles" not in pdf.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sample = pdf.sample(n=min(50_000, len(pdf)), random_state=42)
    ax.scatter(
        sample["NonStopMiles"], sample[target_col],
        alpha=0.1, s=2, color="#4CAF50"
    )
    ax.set_xlabel("Non-Stop Miles")
    ax.set_ylabel("Average Fare ($)")
    ax.set_title("Distance vs Fare")
    plt.tight_layout()
    fig.savefig(eda_dir / "distance_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: distance_vs_fare.png")


def _plot_carrier_fares(pdf, target_col: str, eda_dir: Path) -> None:
    # Top 15 carriers by frequency
    top_carriers = pdf["Carrier"].value_counts().nlargest(15).index.tolist()
    subset = pdf[pdf["Carrier"].isin(top_carriers)]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=subset, x="Carrier", y=target_col,
        order=top_carriers, palette="Set2", ax=ax,
        showfliers=False,
    )
    ax.set_title("Fare Distribution by Carrier (Top 15)")
    ax.set_xlabel("Carrier")
    ax.set_ylabel("Average Fare ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(eda_dir / "carrier_fares.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: carrier_fares.png")


def _plot_correlation_heatmap(pdf, eda_dir: Path) -> None:
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in pdf.columns]
    if TARGET_COLUMN in pdf.columns:
        numeric_cols.append(TARGET_COLUMN)

    if len(numeric_cols) < 2:
        return

    corr = pdf[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, ax=ax, linewidths=0.5,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(eda_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: correlation_heatmap.png")


def _plot_missing_values(df: pl.DataFrame, eda_dir: Path) -> None:
    null_counts = {col: df[col].null_count() for col in df.columns}
    null_pct = {k: v / len(df) * 100 for k, v in null_counts.items() if v > 0}

    if not null_pct:
        logger.info("  No missing values found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cols = list(null_pct.keys())
    vals = list(null_pct.values())
    ax.barh(cols, vals, color="#E91E63")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Values by Column")
    plt.tight_layout()
    fig.savefig(eda_dir / "missing_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: missing_values.png")


def _plot_feature_cardinality(df: pl.DataFrame, eda_dir: Path) -> None:
    cardinalities = {}
    for col in df.columns:
        cardinalities[col] = df[col].n_unique()

    # Sort by cardinality
    sorted_cards = sorted(cardinalities.items(), key=lambda x: x[1], reverse=True)
    cols = [x[0] for x in sorted_cards[:20]]
    vals = [x[1] for x in sorted_cards[:20]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(cols[::-1], vals[::-1], color="#9C27B0")
    ax.set_xlabel("Unique Values")
    ax.set_title("Feature Cardinality (Top 20)")
    plt.tight_layout()
    fig.savefig(eda_dir / "feature_cardinality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: feature_cardinality.png")


def _plot_hhi_distribution(pdf, eda_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pdf["Market_HHI"].dropna(), bins=50, color="#00BCD4", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Herfindahl-Hirschman Index")
    ax.set_ylabel("Count")
    ax.set_title("Market Concentration (HHI) Distribution")
    ax.axvline(x=2500, color="red", linestyle="--", label="Highly concentrated threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(eda_dir / "hhi_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: hhi_distribution.png")


def _plot_top_routes(df: pl.DataFrame, target_col: str, eda_dir: Path) -> None:
    top_routes = (
        df.group_by("ODPairID")
        .agg([
            pl.col(target_col).mean().alias("avg_fare"),
            pl.col(target_col).count().alias("n_records"),
        ])
        .filter(pl.col("n_records") >= 10)
        .sort("avg_fare", descending=True)
        .head(20)
    )

    pdf = top_routes.to_pandas()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(
        [str(x) for x in pdf["ODPairID"].values[::-1]],
        pdf["avg_fare"].values[::-1],
        color="#FF5722",
    )
    ax.set_xlabel("Average Fare ($)")
    ax.set_title("Top 20 Routes by Average Fare (min 10 records)")
    plt.tight_layout()
    fig.savefig(eda_dir / "top_routes_by_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: top_routes_by_fare.png")


def _plot_binary_features(pdf, target_col: str, eda_dir: Path) -> None:
    binary_cols = [c for c in BINARY_COLUMNS if c in pdf.columns]
    if not binary_cols:
        return

    n = len(binary_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, binary_cols):
        sns.boxplot(data=pdf, x=col, y=target_col, ax=ax, palette="Set3", showfliers=False)
        ax.set_title(f"{target_col} by {col}")

    plt.tight_layout()
    fig.savefig(eda_dir / "binary_features_vs_fare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: binary_features_vs_fare.png")


def _save_summary_stats(df: pl.DataFrame, target_col: str, eda_dir: Path) -> None:
    """Save summary statistics as CSV."""
    stats = df.describe()
    stats.write_csv(eda_dir / "summary_statistics.csv")
    logger.info("  Saved: summary_statistics.csv")
