"""Streamlit dashboard for airline fare prediction and revenue management."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Airline Revenue Management Lab",
    page_icon="✈️",
    layout="wide",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_project_root() -> Path:
    """Find project root from this file's location."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]


ROOT = find_project_root()


def main() -> None:
    st.title("✈️ Airline Revenue Management Lab")
    st.markdown(
        "ML-powered fare prediction and revenue simulation using public airline market data."
    )

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["Overview", "EDA Summary", "Model Comparison", "Revenue Simulator", "Feature Importance"],
    )

    if page == "Overview":
        render_overview()
    elif page == "EDA Summary":
        render_eda()
    elif page == "Model Comparison":
        render_model_comparison()
    elif page == "Revenue Simulator":
        render_revenue_simulator()
    elif page == "Feature Importance":
        render_feature_importance()


def render_overview() -> None:
    st.header("📊 Project Overview")

    col1, col2, col3 = st.columns(3)

    # Try loading processed data
    parquet_path = ROOT / "data" / "processed" / "airline_fares.parquet"
    if parquet_path.exists():
        df = load_data(str(parquet_path))
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Features", str(len(df.columns)))
        if "Average_Fare" in df.columns:
            col3.metric("Avg Fare", f"${df['Average_Fare'].mean():.2f}")

        st.subheader("Dataset Sample")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("No processed data found. Run the pipeline first.")

    st.markdown("---")
    st.markdown(
        """
        > **Disclaimer:** This project uses public airline market fare data to approximate
        > route-level fare and revenue-management concepts. Real airline RM systems rely on
        > proprietary booking curves, inventory controls, fare-class availability,
        > cancellations/no-shows, and unconstrained demand estimates.
        """
    )


def render_eda() -> None:
    st.header("📈 Exploratory Data Analysis")

    eda_dir = ROOT / "data" / "reports" / "eda"
    if not eda_dir.exists():
        st.warning("No EDA reports found. Run 'airline-rm eda' first.")
        return

    # Display all plots
    plots = sorted(eda_dir.glob("*.png"))
    if not plots:
        st.info("No EDA plots generated yet.")
        return

    for plot in plots:
        name = plot.stem.replace("_", " ").title()
        st.subheader(name)
        st.image(str(plot), use_container_width=True)

    # Summary stats
    stats_path = eda_dir / "summary_statistics.csv"
    if stats_path.exists():
        st.subheader("Summary Statistics")
        st.dataframe(load_csv(str(stats_path)), use_container_width=True)


def render_model_comparison() -> None:
    st.header("🏆 Model Comparison")

    model_dir = ROOT / "models"
    if not model_dir.exists():
        st.warning("No trained models found. Run training first.")
        return

    # Load all metrics
    metrics_files = sorted(model_dir.glob("*_metrics.json"))
    if not metrics_files:
        st.info("No model metrics found.")
        return

    rows = []
    for mf in metrics_files:
        name = mf.stem.replace("_metrics", "")
        with open(mf) as f:
            m = json.load(f)
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("rmse")

    # Styled table
    st.dataframe(
        df.style.highlight_min(subset=["mae", "rmse", "mape", "smape"], color="#90EE90")
        .highlight_max(subset=["r2"], color="#90EE90")
        .format("{:.4f}"),
        use_container_width=True,
    )

    # Bar chart comparison
    st.subheader("RMSE Comparison")
    st.bar_chart(df["rmse"])

    # Training curves for deep models
    for curve_file in model_dir.glob("*_training_curves.png"):
        name = curve_file.stem.replace("_training_curves", "")
        st.subheader(f"{name} Training Curves")
        st.image(str(curve_file), use_container_width=True)


def render_revenue_simulator() -> None:
    st.header("💰 Revenue Simulator")

    revenue_dir = ROOT / "data" / "reports" / "revenue"
    scenarios_path = revenue_dir / "revenue_scenarios.csv"

    if not scenarios_path.exists():
        st.warning("No revenue simulation results. Run 'airline-rm revenue-sim' first.")

        # Interactive simulator
        st.subheader("Quick Simulator")
        base_fare = st.slider("Base Fare ($)", 50, 1000, 250)
        base_demand = st.slider("Base Demand (passengers)", 100, 50000, 5000)
        elasticity = st.slider("Price Elasticity", -3.0, 0.0, -1.2, step=0.1)

        scenarios = np.arange(-0.20, 0.25, 0.05)
        results = []
        for pct in scenarios:
            new_fare = base_fare * (1 + pct)
            new_demand = max(0, base_demand * (1 + elasticity * pct))
            revenue = new_fare * new_demand
            results.append({
                "Price Change": f"{pct:+.0%}",
                "New Fare": f"${new_fare:.0f}",
                "New Demand": f"{new_demand:,.0f}",
                "Revenue": f"${revenue:,.0f}",
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True)
        return

    df = load_csv(str(scenarios_path))
    st.dataframe(df.head(100), use_container_width=True)

    # Route selector
    if "ODPairID" in df.columns:
        routes = df["ODPairID"].unique()[:50]
        selected = st.selectbox("Select Route", routes)
        route_data = df[df["ODPairID"] == selected]
        st.subheader(f"Route {selected} — Revenue Scenarios")
        st.bar_chart(route_data.set_index("scenario")["new_revenue"])


def render_feature_importance() -> None:
    st.header("🔍 Feature Importance")

    model_dir = ROOT / "models"
    imp_files = sorted(model_dir.glob("*_feature_importance.json"))

    if not imp_files:
        st.warning("No feature importance data found.")
        return

    for imp_file in imp_files:
        model_name = imp_file.stem.replace("_feature_importance", "")
        with open(imp_file) as f:
            imp = json.load(f)

        st.subheader(f"{model_name.upper()}")
        df = pd.DataFrame(
            list(imp.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False).head(20)
        st.bar_chart(df.set_index("Feature"))

    # SHAP plots
    explain_dir = ROOT / "data" / "reports" / "explainability"
    if explain_dir.exists():
        for img in explain_dir.glob("*.png"):
            st.subheader(img.stem.replace("_", " ").title())
            st.image(str(img), use_container_width=True)


if __name__ == "__main__":
    main()
