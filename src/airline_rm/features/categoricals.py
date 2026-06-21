"""Categorical encoding utilities — frequency and target encoding."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger


def frequency_encode(
    df: pl.DataFrame,
    columns: list[str],
    suffix: str = "_freq_enc",
) -> pl.DataFrame:
    """Add frequency encoding for specified columns.

    Computes value_counts / total_rows for each column.
    """
    for col in columns:
        if col not in df.columns:
            continue
        freq = (
            df.group_by(col)
            .agg(pl.len().alias("_count"))
            .with_columns((pl.col("_count") / len(df)).alias(f"{col}{suffix}"))
            .drop("_count")
        )
        df = df.join(freq, on=col, how="left")
        logger.debug(f"Frequency encoded: {col} → {col}{suffix}")

    return df


def target_encode_cv(
    df: pl.DataFrame,
    column: str,
    target: str,
    n_folds: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
    suffix: str = "_target_enc",
) -> pl.DataFrame:
    """Target encoding with cross-validation fold isolation to prevent leakage.

    For each fold, the encoding is computed from the OTHER folds only.
    Uses Bayesian smoothing: encoded = (count * mean_cat + smoothing * global_mean) / (count + smoothing)
    """
    result_col = f"{column}{suffix}"
    global_mean = df[target].mean()

    # Assign fold indices
    n = len(df)
    rng = np.random.default_rng(seed)
    fold_ids = rng.integers(0, n_folds, size=n)
    df = df.with_columns(pl.Series("_fold", fold_ids))

    # Initialize result column
    encoded_values = np.full(n, global_mean, dtype=np.float32)

    for fold in range(n_folds):
        train_mask = df["_fold"] != fold
        val_mask = df["_fold"] == fold

        train_df = df.filter(train_mask)

        # Compute target stats from training folds
        stats = (
            train_df.group_by(column)
            .agg([
                pl.col(target).mean().alias("_mean"),
                pl.col(target).count().alias("_count"),
            ])
        )

        # Bayesian smoothing
        stats = stats.with_columns(
            (
                (pl.col("_count") * pl.col("_mean") + smoothing * global_mean)
                / (pl.col("_count") + smoothing)
            ).alias("_smoothed")
        )

        # Map to validation fold
        val_df = df.filter(val_mask)
        val_mapped = val_df.select(column).join(
            stats.select([column, "_smoothed"]),
            on=column,
            how="left",
        )

        val_indices = np.where(fold_ids == fold)[0]
        values = val_mapped["_smoothed"].fill_null(global_mean).to_numpy()
        encoded_values[val_indices] = values.astype(np.float32)

    df = df.with_columns(pl.Series(result_col, encoded_values)).drop("_fold")
    logger.debug(f"Target encoded (CV): {column} → {result_col}")

    return df
