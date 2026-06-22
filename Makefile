.PHONY: help install lint format typecheck test test-cov clean download inspect build-parquet eda build-features train evaluate explain export-tableau dashboard

SHELL := /bin/zsh
export PATH := /Users/antonio/miniconda3/bin:$(PATH)
POETRY := /Users/antonio/miniconda3/bin/poetry
PYTHON := $(POETRY) run python
CLI := $(POETRY) run airline-rm

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === Setup ===

install: ## Install all dependencies with Poetry
	poetry install --with dev,dashboard

install-base: ## Install only core dependencies
	poetry install

# === Code Quality ===

lint: ## Run ruff linter
	poetry run ruff check src/ tests/

format: ## Format code with ruff
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

typecheck: ## Run mypy type checking
	poetry run mypy src/

# === Testing ===

test: ## Run all tests
	poetry run pytest tests/ -v

test-fast: ## Run tests excluding slow markers
	poetry run pytest tests/ -v -m "not slow"

test-cov: ## Run tests with coverage
	poetry run pytest tests/ -v --cov=airline_rm --cov-report=html --cov-report=term

# === Pipeline Commands ===

download: ## Download dataset from Kaggle
	$(CLI) download

inspect: ## Inspect raw dataset files
	$(CLI) inspect

build-parquet: ## Convert raw CSV to Parquet
	$(CLI) build-parquet

eda: ## Run exploratory data analysis
	$(CLI) eda

build-features: ## Build feature dataset
	$(CLI) build-features

# === Training ===

train-baseline: ## Train baseline models
	$(CLI) train --model baseline

train-xgboost: ## Train XGBoost model
	$(CLI) train --model xgboost

train-lightgbm: ## Train LightGBM model
	$(CLI) train --model lightgbm

train-catboost: ## Train CatBoost model
	$(CLI) train --model catboost

train-lstm: ## Train LSTM attention model
	$(CLI) train --model lstm_attention

train-transformer: ## Train Transformer model
	$(CLI) train --model transformer

train-all: ## Train all models sequentially
	$(CLI) train --model baseline
	$(CLI) train --model xgboost
	$(CLI) train --model catboost
	$(CLI) train --model lstm_attention
	$(CLI) train --model transformer

# === Evaluation & Explainability ===

evaluate: ## Evaluate and compare all trained models
	$(CLI) evaluate

explain: ## Run explainability analysis
	$(CLI) explain

# === Exports ===

export-tableau: ## Export data for Tableau
	$(CLI) export-tableau

# === Dashboard ===

dashboard: ## Launch Streamlit dashboard
	$(POETRY) run streamlit run src/airline_rm/dashboard.py

# === Cleanup ===

clean: ## Remove generated artifacts
	rm -rf data/interim/* data/processed/* data/features/* data/reports/* data/exports/*
	rm -rf models/
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Remove everything including raw data
	rm -rf data/raw/* data/metadata/*
