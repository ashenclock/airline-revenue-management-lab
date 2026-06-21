"""Project path resolution — all paths relative to project root."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file to find the directory containing pyproject.toml.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: 4 levels up from src/airline_rm/utils/paths.py
    return current.parents[3]


ROOT = project_root()


def data_dir(subdir: str = "") -> Path:
    """Return path under data/, creating it if needed."""
    p = ROOT / "data" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    return data_dir("raw")


def interim_dir() -> Path:
    return data_dir("interim")


def processed_dir() -> Path:
    return data_dir("processed")


def features_dir() -> Path:
    return data_dir("features")


def reports_dir(subdir: str = "") -> Path:
    return data_dir(f"reports/{subdir}" if subdir else "reports")


def exports_dir() -> Path:
    return data_dir("exports")


def metadata_dir() -> Path:
    return data_dir("metadata")


def models_dir() -> Path:
    p = ROOT / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def configs_dir() -> Path:
    return ROOT / "configs"
