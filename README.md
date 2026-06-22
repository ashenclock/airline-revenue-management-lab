# ✈️ Airline Revenue Management Lab

> Production-quality ML pipeline for **airline market fare prediction** and **revenue management simulation**, built on public DOT/BTS market data.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Revenue Simulation](#-revenue-simulation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [Memory-Aware Pipeline](#-memory-aware-pipeline)
- [Explainability](#-explainability)
- [Tableau Exports](#-tableau-exports)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🎯 Business Problem

Airline revenue management (RM) is the practice of **optimizing pricing and seat inventory** to maximize revenue. At its core, RM answers: *"What fare should we set on each route, for each market segment, to maximize total revenue?"*

This project demonstrates:
- **Fare prediction**: Predicting average market fares from route, carrier, and competition features
- **Revenue simulation**: Estimating revenue impact of pricing changes using price elasticity models
- **Model comparison**: Benchmarking baselines, gradient-boosted trees, and deep learning architectures
- **Explainability**: Understanding which factors drive fare levels

---

## 📊 Dataset

**Source**: [Airline Market Fare Prediction Data](https://www.kaggle.com/datasets/orvile/airline-market-fare-prediction-data) (Kaggle)
**Origin**: U.S. Department of Transportation — Bureau of Transportation Statistics (DB1B / T-100)

| Property | Value |
|----------|-------|
| Rows | ~1,581,278 |
| Columns | 26 (20 after removing pre-computed frequency encodings) |
| Target | `Average_Fare` (market average fare in USD) |
| Size | ~150–200 MB (CSV), ~60 MB (Parquet) |

### Key Features

| Feature | Description |
|---------|-------------|
| `NonStopMiles` | Great-circle distance between origin and destination |
| `Carrier` | Airline carrier code (AA, UA, DL, WN, etc.) |
| `ODPairID` | Origin-destination route identifier |
| `Pax` | Total passengers on route |
| `Market_HHI` | Herfindahl-Hirschman Index (market concentration) |
| `LCC_Comp` | Low-cost carrier competition present |
| `Circuity` | Ratio of actual distance to nonstop distance |
| `Market_share` | Carrier's share of route traffic |

### Limitations

This is **cross-sectional** data — no Year/Quarter columns. This means:
- No true time-series forecasting capability
- Deep learning models operate on route-level grouped records rather than temporal sequences
- Revenue simulation uses constant elasticity assumptions rather than real booking curves

---

## 🔬 Methodology

### Pipeline Stages

```
Download → Inspect → Clean → Parquet → Features → Train → Evaluate → Explain → Export
```

### Feature Engineering

| Category | Features |
|----------|----------|
| **Route-level** | Avg/std fare on route, carrier count, total passengers, competition level |
| **Carrier-level** | Avg fare, route count, avg distance, passenger share |
| **Market-level** | Fare per mile, circuity premium, hub origin/destination flags |
| **Frequency encoding** | Recomputed from data (dropped pre-computed columns to avoid leakage) |
| **Target encoding** | Cross-validated with Bayesian smoothing to prevent data leakage |

### Data Validation

- ✅ No negative fares
- ✅ No impossible distances
- ✅ No target leakage from pre-computed encodings
- ✅ Duplicate detection and removal
- ✅ Missing value analysis

---

## 🏆 Models & Results

### Phase 1: Baselines

| Model | MAE | RMSE | R² |
### 📊 1. Quantitative Evaluation
| Model               | MAE    | RMSE   | MAPE% | SMAPE% | R²     |
|---------------------|--------|--------|-------|--------|--------|
| **XGBoost**         | **1.68**| **3.01**| **0.83**| **0.83**| **0.9983**|
| CatBoost            | 3.08   | 4.58   | 1.49  | 1.49   | 0.9961 |
| Transformer         | 26.25  | 45.60  | 12.84 | 12.49  | 0.6996 |
| LSTM w/ Attention   | 48.74  | 76.18  | 26.21 | 23.94  | 0.1616 |
| Global Mean Base    | 53.35  | 72.87  | 27.48 | 24.00  | 0.0000 |
| Carrier Mean Base   | 64.21  | 85.57  | 32.61 | 29.12  | -0.3787|
| Route Mean Base     | 73.87  | 99.51  | 36.70 | 32.24  | -0.8647|
| Ridge Baseline      | 76.54  | 102.31 | 38.06 | 34.10  | -0.9710|

> 📝 Results will be filled after running the pipeline. Run `make train-all && make evaluate`.

### Deep Learning Architectures

**LSTM + Attention Pooling**: LSTM encoder processes route-level observations, then attention pooling learns which observations are most informative for fare prediction.

**Gated Attention Pooling**: Extends attention with a sigmoid gating mechanism:
```
α = softmax(W_a · h)      # attention weights
g = σ(W_g · h)             # gate values
output = Σ(α · g · h)      # gated weighted sum
```

**Transformer Encoder**: Self-attention across route observations with positional encoding, followed by attention pooling and a regression head.

---

## 💰 Revenue Simulation

> **Important**: This is a simplified educational simulation. See [Disclaimer](#-disclaimer).

Uses a constant-elasticity demand model:

```
new_demand = base_demand × (1 + elasticity × price_change%)
revenue = new_fare × new_demand
```

### Price Scenarios
| Scenario | Price Change |
|----------|-------------|
| Aggressive discount | -20% |
| Moderate discount | -10% |
| Small discount | -5% |
| Current pricing | 0% |
| Small premium | +5% |
| Moderate premium | +10% |
| Aggressive premium | +20% |

Default elasticity: **-1.2** (configurable in `configs/models.toml`)

---

## 📁 Project Structure

```
airline-revenue-management-lab/
├── README.md
├── pyproject.toml          # Poetry config, dependencies, CLI entry points
├── Makefile                # Pipeline commands
├── .gitignore
├── .env.example
├── .pre-commit-config.yaml
├── LICENSE
├── configs/
│   ├── default.toml        # Dataset, paths, memory, splits
│   └── models.toml         # Hyperparameters for all models
├── data/
│   ├── raw/                # Downloaded CSVs (git-ignored)
│   ├── processed/          # Cleaned Parquet files
│   ├── features/           # Feature-engineered dataset
│   ├── reports/eda/        # EDA plots and stats
│   ├── exports/            # Tableau-ready CSVs
│   └── metadata/           # Dataset manifest
├── notebooks/              # Lightweight visualization notebooks
├── src/airline_rm/
│   ├── cli.py              # Typer CLI entry point
│   ├── dashboard.py        # Streamlit app
│   ├── config/settings.py  # Pydantic settings from TOML
│   ├── data/               # Download, inspect, clean, parquet
│   ├── features/           # Categoricals, route/carrier features, sequences
│   ├── models/             # Baselines, XGB, LGB, CatBoost, LSTM, Transformer
│   ├── explainability/     # Feature importance, SHAP
│   ├── visualization/      # EDA plots, Tableau exports
│   └── utils/              # Logging, memory, paths, parallelism
├── tests/                  # Unit tests (pytest)
└── models/                 # Saved model artifacts (git-ignored)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- macOS with Apple Silicon (M1/M2) recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/airline-revenue-management-lab.git
cd airline-revenue-management-lab

# Install dependencies
poetry install --with dev,dashboard

# Copy environment template
cp .env.example .env
```

### Running the Pipeline

```bash
# 1. Download dataset (requires Kaggle credentials or manual download)
make download

# 2. Inspect raw data
make inspect

# 3. Convert to Parquet
make build-parquet

# 4. Run EDA
make eda

# 5. Build features
make build-features

# 6. Train models (sequentially)
make train-baseline
make train-xgboost
make train-lightgbm
make train-catboost
make train-lstm
make train-transformer

# 7. Compare models
make evaluate

# 8. Explainability
make explain

# 9. Export for Tableau
make export-tableau

# 10. Launch dashboard
make dashboard
```

### Running on Mac M1 (8GB RAM)

The entire pipeline is designed for constrained hardware:
- Polars lazy scans for CSV inspection
- Chunked reading with configurable batch sizes
- Memory guards that raise `MemoryError` before OOM
- psutil-based monitoring throughout the pipeline
- Conservative parallelism defaults (`n_jobs=2`)
- Small deep learning models (< 1M parameters)
- No mixed precision (float32 only — MPS stability)

---

## 📟 CLI Reference

```bash
poetry run airline-rm download        # Download dataset
poetry run airline-rm inspect         # Inspect raw files
poetry run airline-rm build-parquet   # CSV → Parquet
poetry run airline-rm eda             # EDA plots
poetry run airline-rm build-features  # Feature engineering
poetry run airline-rm train --model xgboost
poetry run airline-rm train --model lightgbm
poetry run airline-rm train --model catboost
poetry run airline-rm train --model lstm_attention
poetry run airline-rm train --model transformer
poetry run airline-rm evaluate        # Compare all models
poetry run airline-rm explain         # Feature importance + SHAP
poetry run airline-rm export-tableau  # Tableau CSVs
poetry run airline-rm revenue-sim     # Revenue simulation
poetry run airline-rm dashboard       # Streamlit app
```

---

## 🧠 Memory-Aware Pipeline

| Stage | Memory Strategy |
|-------|----------------|
| CSV inspection | Polars lazy scan (zero copy) |
| CSV → Parquet | Full load (~200MB) with memory guard |
| Feature building | Polars in-memory with downcasting |
| Tree training | `tree_method="hist"`, CPU, `n_jobs=2` |
| Deep learning | Small models, batch_size=64, MPS/CPU |
| SHAP | Subsampled to 5,000 rows max |

All operations check `psutil.virtual_memory()` before proceeding and raise `MemoryError` with context if the limit (default 5GB) is exceeded.

---

## 🔍 Explainability

### Tree Models
- **Built-in feature importance** (gain/split)
- **Permutation importance** on validation sample
- **SHAP TreeExplainer** on memory-safe subsample (max 5,000 rows)

### Deep Models
- **Attention weight visualization** for sample sequences
- Documented limitations of attention as explanation

---

## 📊 Tableau Exports

The pipeline generates Tableau-ready CSVs:

| File | Contents |
|------|----------|
| `tableau_route_summary.csv` | Route-level fare/demand/competition stats |
| `tableau_feature_importance.csv` | Feature importance across all models |
| `tableau_revenue_scenarios.csv` | Revenue under different pricing scenarios |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **Package Manager** | Poetry |
| **Data** | Polars, Pandas, PyArrow, NumPy |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch (MPS/CPU) |
| **CLI** | Typer |
| **Config** | Pydantic + TOML |
| **Logging** | Loguru |
| **Visualization** | Matplotlib, Seaborn, Streamlit, Plotly |
| **Explainability** | SHAP |
| **Testing** | pytest |
| **Linting** | ruff, mypy |

---

## 🔮 Future Improvements

- [ ] Add temporal data (DB1B quarterly) for true time-series forecasting
- [ ] Implement fare-class level prediction
- [ ] Add booking curve simulation
- [ ] MLflow experiment tracking
- [ ] Hyperparameter tuning with Optuna
- [ ] ONNX model export for serving
- [ ] Containerized pipeline with Docker
- [ ] CI/CD with GitHub Actions
- [ ] MLX backend for Apple Silicon optimization
- [ ] DuckDB for local analytical queries

---

## ⚠️ Disclaimer

This project uses **public airline market fare data** to approximate route-level fare and revenue-management concepts.

**Real airline revenue management systems** rely on:
- Proprietary booking curves and forecasting models
- Fare-class inventory controls (EMSR, bid prices)
- Cancellation and no-show models
- Unconstrained demand estimation
- Network-level optimization across connecting flights
- Real-time competitive pricing data

Therefore, this project combines public fare/market data with **transparent synthetic assumptions** for revenue simulation. It is designed as a **portfolio project** demonstrating ML engineering, not as a production RM system.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ☕ and 🧠 by Antonio*
