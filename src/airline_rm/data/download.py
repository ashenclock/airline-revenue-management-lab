"""Dataset downloader — supports kagglehub and manual fallback."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from airline_rm.config.settings import Settings, load_settings
from airline_rm.utils.paths import metadata_dir, raw_dir


def download_dataset(settings: Settings | None = None) -> Path:
    """Download the airline fare dataset from Kaggle.

    Strategy:
    1. Try kagglehub (simplest — uses ~/.kaggle/kaggle.json or env vars).
    2. If no credentials, print manual download instructions.

    Returns:
        Path to the directory containing downloaded files.
    """
    if settings is None:
        settings = load_settings()

    slug = settings.dataset.slug
    dest = raw_dir()

    # Check if already downloaded
    existing = list(dest.glob("*.csv"))
    if existing:
        logger.info(f"Dataset already exists in {dest} ({len(existing)} CSV files). Skipping.")
        return dest

    logger.info(f"Downloading dataset: {slug}")

    try:
        import kagglehub

        cache_path = kagglehub.dataset_download(slug)
        cache_path = Path(cache_path)
        logger.info(f"Downloaded to cache: {cache_path}")

        # Copy files to data/raw/
        for f in cache_path.iterdir():
            if f.is_file():
                target = dest / f.name
                shutil.copy2(f, target)
                logger.info(f"  Copied: {f.name} → {target}")

    except Exception as e:
        logger.warning(f"kagglehub download failed: {e}")
        logger.info("=" * 60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.info(f"  1. Go to: https://www.kaggle.com/datasets/{slug}")
        logger.info("  2. Click 'Download' to get the ZIP file.")
        logger.info(f"  3. Extract CSV files into: {dest}")
        logger.info("")
        logger.info("  Or set Kaggle credentials:")
        logger.info("    export KAGGLE_USERNAME=your_username")
        logger.info("    export KAGGLE_KEY=your_api_key")
        logger.info("=" * 60)
        return dest

    # Write manifest
    _write_manifest(dest, slug)
    return dest


def _write_manifest(data_dir: Path, slug: str) -> None:
    """Write dataset metadata manifest."""
    meta_dir = metadata_dir()
    files_info = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file():
            files_info.append({
                "name": f.name,
                "size_bytes": f.stat().st_size,
                "size_mb": round(f.stat().st_size / (1024**2), 2),
            })

    manifest = {
        "dataset_slug": slug,
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "kaggle",
        "files": files_info,
        "total_files": len(files_info),
    }

    manifest_path = meta_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")
