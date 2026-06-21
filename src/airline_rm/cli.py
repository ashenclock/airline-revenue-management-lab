"""Typer CLI — unified entry point for all pipeline commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from airline_rm.config.settings import load_settings
from airline_rm.utils.logging import setup_logging

app = typer.Typer(
    name="airline-rm",
    help="Airline Revenue Management Lab — ML pipeline for fare prediction.",
    add_completion=False,
)


@app.callback()
def main(log_level: str = typer.Option("INFO", help="Logging level")) -> None:
    """Configure logging for all commands."""
    setup_logging(log_level)


@app.command()
def download() -> None:
    """Download the airline fare dataset from Kaggle."""
    from airline_rm.data.download import download_dataset

    settings = load_settings()
    path = download_dataset(settings)
    typer.echo(f"✅ Dataset downloaded to: {path}")


@app.command()
def inspect() -> None:
    """Inspect raw dataset files (schema, dtypes, nulls, stats)."""
    from airline_rm.data.inspect import inspect_dataset

    report = inspect_dataset()
    if report:
        typer.echo(f"✅ Inspection complete. {len(report)} file(s) analyzed.")
    else:
        typer.echo("❌ No files to inspect. Run 'download' first.")
        raise typer.Exit(code=1)


@app.command(name="build-parquet")
def build_parquet() -> None:
    """Convert raw CSV to cleaned compressed Parquet."""
    from airline_rm.data.parquet_builder import build_parquet as _build

    path = _build()
    typer.echo(f"✅ Parquet built: {path}")


@app.command()
def eda() -> None:
    """Run exploratory data analysis and save plots."""
    import polars as pl
    from airline_rm.visualization.eda_plots import run_eda
    from airline_rm.utils.paths import processed_dir

    parquet_path = processed_dir() / "airline_fares.parquet"
    if not parquet_path.exists():
        typer.echo("❌ Processed data not found. Run 'build-parquet' first.")
        raise typer.Exit(code=1)

    df = pl.read_parquet(parquet_path)
    eda_dir = run_eda(df)
    typer.echo(f"✅ EDA complete. Plots saved to: {eda_dir}")


@app.command(name="build-features")
def build_features() -> None:
    """Build feature dataset from processed Parquet."""
    from airline_rm.features.build_features import build_features as _build

    path = _build()
    typer.echo(f"✅ Features built: {path}")


@app.command()
def train(
    model: str = typer.Option(
        ...,
        help="Model to train: baseline, xgboost, lightgbm, catboost, lstm_attention, transformer",
    ),
) -> None:
    """Train a specified model."""
    from airline_rm.models.train import train_model

    valid_models = ["baseline", "xgboost", "lightgbm", "catboost", "lstm_attention", "transformer"]
    if model not in valid_models:
        typer.echo(f"❌ Unknown model: {model}. Choose from: {valid_models}")
        raise typer.Exit(code=1)

    metrics = train_model(model)
    typer.echo(f"✅ Model '{model}' trained successfully.")
    if isinstance(metrics, dict) and "rmse" in metrics:
        typer.echo(f"   RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")


@app.command()
def evaluate() -> None:
    """Compare all trained models."""
    from airline_rm.models.evaluate import compare_models, print_comparison_table
    from airline_rm.utils.paths import models_dir

    comparison = compare_models(models_dir())
    if not comparison:
        typer.echo("❌ No trained models found. Run 'train' first.")
        raise typer.Exit(code=1)

    print_comparison_table(comparison)
    typer.echo(f"✅ Compared {len(comparison)} models.")


@app.command()
def explain() -> None:
    """Run explainability analysis (feature importance, SHAP)."""
    import json
    from airline_rm.explainability.feature_importance import plot_feature_importance
    from airline_rm.utils.paths import models_dir

    model_dir = models_dir()
    for imp_file in model_dir.glob("*_feature_importance.json"):
        model_name = imp_file.stem.replace("_feature_importance", "")
        with open(imp_file) as f:
            imp = json.load(f)
        plot_feature_importance(imp, model_name)
        typer.echo(f"  📊 Plotted importance for: {model_name}")

    typer.echo("✅ Explainability analysis complete.")


@app.command(name="export-tableau")
def export_tableau() -> None:
    """Export data for Tableau visualization."""
    import polars as pl
    from airline_rm.visualization.tableau_exports import export_all
    from airline_rm.utils.paths import processed_dir

    parquet_path = processed_dir() / "airline_fares.parquet"
    if not parquet_path.exists():
        typer.echo("❌ Processed data not found. Run 'build-parquet' first.")
        raise typer.Exit(code=1)

    df = pl.read_parquet(parquet_path)
    paths = export_all(df)
    typer.echo(f"✅ Exported {len(paths)} files for Tableau.")


@app.command(name="revenue-sim")
def revenue_sim() -> None:
    """Run revenue management simulation."""
    import polars as pl
    from airline_rm.models.revenue_simulator import run_revenue_simulation
    from airline_rm.utils.paths import processed_dir

    parquet_path = processed_dir() / "airline_fares.parquet"
    if not parquet_path.exists():
        typer.echo("❌ Processed data not found. Run 'build-parquet' first.")
        raise typer.Exit(code=1)

    df = pl.read_parquet(parquet_path)
    scenarios, ranking = run_revenue_simulation(df)
    typer.echo(f"✅ Revenue simulation complete: {len(scenarios)} scenario results.")


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


if __name__ == "__main__":
    app()
