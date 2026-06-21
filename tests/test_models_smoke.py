"""Smoke tests for all model types on tiny synthetic data."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from airline_rm.models.baseline import MeanBaseline, GroupMeanBaseline, RidgeBaseline


def test_mean_baseline(sample_df):
    """MeanBaseline should predict the global mean."""
    model = MeanBaseline().fit(sample_df)
    preds = model.predict(sample_df)
    assert len(preds) == len(sample_df)
    assert abs(preds[0] - sample_df["Average_Fare"].mean()) < 0.01


def test_group_mean_baseline(sample_df):
    """GroupMeanBaseline should predict per-group means."""
    model = GroupMeanBaseline("Carrier").fit(sample_df)
    preds = model.predict(sample_df)
    assert len(preds) == len(sample_df)
    assert np.all(np.isfinite(preds))


def test_ridge_baseline(sample_df, feature_cols):
    """RidgeBaseline should produce finite predictions."""
    model = RidgeBaseline(feature_cols=feature_cols)
    model.fit(sample_df)
    preds = model.predict(sample_df)
    assert len(preds) == len(sample_df)
    assert np.all(np.isfinite(preds))


def test_xgboost_smoke(sample_df, feature_cols):
    """XGBoost should train and predict without errors."""
    from airline_rm.models.xgboost_model import XGBoostModel

    X = sample_df.select(feature_cols).to_numpy().astype(np.float32)
    y = sample_df["Average_Fare"].to_numpy().astype(np.float32)

    model = XGBoostModel({"n_estimators": 10, "max_depth": 3, "early_stopping_rounds": 5})
    model.fit(X[:150], y[:150], X[150:], y[150:], feature_names=feature_cols)
    preds = model.predict(X[150:])
    assert len(preds) == 50
    assert np.all(np.isfinite(preds))


def test_lstm_attention_smoke():
    """LSTM with attention should forward pass without errors."""
    from airline_rm.models.torch_sequence_models import LSTMAttentionModel

    model = LSTMAttentionModel(input_dim=5, hidden_dim=16, num_layers=1)
    x = torch.randn(4, 10, 5)  # batch=4, seq=10, features=5
    mask = torch.ones(4, 10)
    pred, attn = model(x, mask)

    assert pred.shape == (4,)
    assert attn.shape == (4, 10)
    assert torch.all(torch.isfinite(pred))


def test_gated_attention_smoke():
    """LSTM with gated attention should forward pass."""
    from airline_rm.models.torch_sequence_models import LSTMAttentionModel

    model = LSTMAttentionModel(
        input_dim=5, hidden_dim=16, num_layers=1, use_gated_attention=True
    )
    x = torch.randn(4, 10, 5)
    mask = torch.ones(4, 10)
    pred, attn = model(x, mask)

    assert pred.shape == (4,)
    assert torch.all(torch.isfinite(pred))


def test_transformer_smoke():
    """Transformer should forward pass without errors."""
    from airline_rm.models.torch_sequence_models import TransformerFareModel

    model = TransformerFareModel(
        input_dim=5, d_model=16, nhead=2, num_encoder_layers=1
    )
    x = torch.randn(4, 10, 5)
    mask = torch.ones(4, 10)
    pred, attn = model(x, mask)

    assert pred.shape == (4,)
    assert torch.all(torch.isfinite(pred))


def test_sequence_dataset(sample_df, feature_cols):
    """RouteSequenceDataset should produce valid tensors."""
    from airline_rm.features.sequence_builder import build_sequence_dataset

    dataset = build_sequence_dataset(sample_df, feature_cols, seq_len=5)
    assert len(dataset) > 0

    x, y, mask = dataset[0]
    assert x.shape == (5, len(feature_cols))
    assert mask.shape == (5,)
    assert torch.isfinite(y)
