"""SHAP analysis with memory-safe subsampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from airline_rm.utils.memory import check_memory_limit, get_memory_usage_gb
from airline_rm.utils.paths import reports_dir


def run_shap_analysis(
    model: Any,
    X_sample: np.ndarray,
    feature_names: list[str],
    model_name: str,
    max_samples: int = 5000,
    output_dir: Path | None = None,
    memory_limit_gb: float = 5.0,
) -> Path | None:
    """Run SHAP TreeExplainer on a memory-safe sample.

    Returns path to output dir, or None if SHAP cannot be run safely.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP analysis.")
        return None

    if output_dir is None:
        output_dir = reports_dir("explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Memory check before SHAP
    current_mem = get_memory_usage_gb()
    if current_mem > memory_limit_gb * 0.7:
        logger.warning(
            f"Memory usage too high for SHAP ({current_mem:.1f} GB). Skipping."
        )
        return None

    # Subsample
    if len(X_sample) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_sample), max_samples, replace=False)
        X_sample = X_sample[idx]

    logger.info(f"Running SHAP analysis on {len(X_sample):,} samples...")

    try:
        # Try TreeExplainer (for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False,
        )
        path = output_dir / f"{model_name}_shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP summary plot saved: {path}")

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
        )
        path_bar = output_dir / f"{model_name}_shap_bar.png"
        plt.savefig(path_bar, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info(f"SHAP bar plot saved: {path_bar}")

        return output_dir

    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        return None
