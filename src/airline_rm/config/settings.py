"""Pydantic-based settings loaded from TOML config files and environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli
from pydantic import BaseModel, Field

from airline_rm.utils.paths import configs_dir


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    slug: str = "orvile/airline-market-fare-prediction-data"
    target_column: str = "Average_Fare"
    drop_columns: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    features: str = "data/features"
    reports: str = "data/reports"
    exports: str = "data/exports"
    metadata: str = "data/metadata"
    models: str = "models"


class MemoryConfig(BaseModel):
    limit_gb: float = 5.0
    chunk_size: int = 100_000


class SplittingConfig(BaseModel):
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


class ParallelConfig(BaseModel):
    n_jobs: int = 2


class LoggingConfig(BaseModel):
    level: str = "INFO"


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Top-level project settings loaded from configs/default.toml."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

class ModelHyperparams(BaseModel):
    """Loaded from configs/models.toml — each key is a model name."""

    baseline: dict[str, Any] = Field(default_factory=dict)
    xgboost: dict[str, Any] = Field(default_factory=dict)
    lightgbm: dict[str, Any] = Field(default_factory=dict)
    catboost: dict[str, Any] = Field(default_factory=dict)
    lstm: dict[str, Any] = Field(default_factory=dict)
    transformer: dict[str, Any] = Field(default_factory=dict)
    revenue: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return the parsed dict."""
    with open(path, "rb") as f:
        return tomli.load(f)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load project settings from default.toml."""
    if config_path is None:
        config_path = configs_dir() / "default.toml"
    if not config_path.exists():
        return Settings()
    data = _load_toml(config_path)
    return Settings(**data)


def load_model_hyperparams(config_path: Path | None = None) -> ModelHyperparams:
    """Load model hyperparameters from models.toml."""
    if config_path is None:
        config_path = configs_dir() / "models.toml"
    if not config_path.exists():
        return ModelHyperparams()
    data = _load_toml(config_path)
    return ModelHyperparams(**data)
