"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from airline_rm.models.evaluate import compute_metrics


def test_perfect_predictions():
    """Perfect predictions should yield 0 error and R²=1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_constant_predictions():
    """Mean prediction should give R²=0."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.full_like(y, y.mean())
    metrics = compute_metrics(y, pred)
    assert abs(metrics["r2"]) < 0.01


def test_mae_computation():
    """MAE should be the mean absolute error."""
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    metrics = compute_metrics(y, pred)
    assert abs(metrics["mae"] - 0.5) < 0.001


def test_rmse_computation():
    """RMSE should be sqrt of mean squared error."""
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])
    metrics = compute_metrics(y, pred)
    assert abs(metrics["rmse"] - 1.0) < 0.001


def test_mape_avoids_division_by_zero():
    """MAPE should handle zero true values gracefully."""
    y = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.5, 1.5, 2.5])
    metrics = compute_metrics(y, pred)
    # MAPE computed only on non-zero values
    assert metrics["mape"] < float("inf")


def test_smape_symmetric():
    """SMAPE should be symmetric."""
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([2.0, 3.0, 4.0])
    m1 = compute_metrics(y, pred)["smape"]
    m2 = compute_metrics(pred, y)["smape"]
    assert abs(m1 - m2) < 0.01


def test_revenue_simulator_formula():
    """Revenue simulator elasticity formula should be correct."""
    from airline_rm.models.revenue_simulator import RevenueSimulator

    sim = RevenueSimulator(elasticity=-1.0, price_scenarios=[0.10])
    # With elasticity=-1.0 and +10% price:
    # new_demand = base * (1 + (-1.0) * 0.10) = base * 0.90
    # So demand drops 10%

    df_test = __import__("polars").DataFrame({
        "ODPairID": [1],
        "Average_Fare": [100.0],
        "Pax": [1000],
        "Carrier": ["AA"],
        "NonStopMiles": [500.0],
    })

    scenarios = sim.simulate_route_revenue(df_test)
    row = scenarios.to_dicts()[0]

    assert abs(row["new_fare"] - 110.0) < 0.01
    assert abs(row["new_demand"] - 900.0) < 0.01
    assert abs(row["new_revenue"] - 99000.0) < 1.0
