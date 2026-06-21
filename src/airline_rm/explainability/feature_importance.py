"""Feature importance analysis for tree models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.inspection import permutation_importance

from airline_rm.utils.paths import reports_dir


def plot_feature_importance(
    importance: dict[str, float],
    model_name: str,
    output_dir: Path | None = None,
    top_n: int = 20,
) -> Path:
    """Plot and save feature importance bar chart."""
    if output_dir is None:
        output_dir = reports_dir("explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Take top N
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_imp][::-1]
    values = [x[1] for x in sorted_imp][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
    ax.barh(names, values, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} — Feature Importance (Top {top_n})")
    plt.tight_layout()

    path = output_dir / f"{model_name}_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {path}")
    return path


def compute_permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    sample_size: int = 10_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute permutation importance on a validation sample.

    Subsamples to `sample_size` to control memory usage.
    """
    if len(X_val) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_val), sample_size, replace=False)
        X_sample = X_val[idx]
        y_sample = y_val[idx]
    else:
        X_sample = X_val
        y_sample = y_val

    logger.info(f"Computing permutation importance on {len(X_sample):,} samples...")
    result = permutation_importance(
        model, X_sample, y_sample,
        n_repeats=n_repeats, random_state=random_state, n_jobs=2,
    )

    imp = dict(zip(feature_names, result.importances_mean.tolist()))
    return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
