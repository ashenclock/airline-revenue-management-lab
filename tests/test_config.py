"""Tests for configuration loading."""

from __future__ import annotations

from airline_rm.config.settings import (
    Settings,
    ModelHyperparams,
    load_settings,
    load_model_hyperparams,
)


def test_settings_defaults():
    """Settings should have sensible defaults."""
    s = Settings()
    assert s.dataset.target_column == "Average_Fare"
    assert s.memory.limit_gb == 5.0
    assert s.splitting.random_seed == 42
    assert s.splitting.train_ratio + s.splitting.val_ratio + s.splitting.test_ratio == 1.0


def test_load_settings():
    """Should load settings from TOML file."""
    s = load_settings()
    assert s.dataset.slug == "orvile/airline-market-fare-prediction-data"
    assert s.memory.chunk_size == 100_000


def test_load_model_hyperparams():
    """Should load model hyperparameters."""
    hp = load_model_hyperparams()
    assert hp.xgboost["tree_method"] == "hist"
    assert hp.lightgbm["num_leaves"] == 63
    assert hp.lstm["hidden_dim"] == 64
    assert hp.transformer["nhead"] == 4
