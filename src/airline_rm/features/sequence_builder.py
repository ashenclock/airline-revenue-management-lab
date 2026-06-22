"""Sequence dataset builder for deep learning models.

Since the dataset has no temporal columns, we build "sequences" by grouping
multiple records for the same ODPairID (route) and treating each carrier's
market observations as elements of a sequence.

This lets us demonstrate attention-based architectures even without time-series data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from loguru import logger

from airline_rm.data.schema import TARGET_COLUMN


class RouteSequenceDataset(Dataset):
    """PyTorch Dataset that groups records by ODPairID into fixed-length sequences.

    Each sequence represents multiple observations for the same route,
    padded/truncated to `seq_len`. Features are the numeric columns,
    target is the average fare for the route.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str = TARGET_COLUMN,
        group_col: str = "ODPairID",
        seq_len: int = 10,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Group by route
        groups = df.group_by(group_col, maintain_order=True)
        self.sequences: list[np.ndarray] = []
        self.targets: list[float] = []
        self.masks: list[np.ndarray] = []

        for _key, group_df in groups:
            features = group_df.select(feature_cols).to_numpy().astype(np.float32)
            features = np.nan_to_num(features, nan=0.0)
            target = group_df[target_col].mean()

            if target is None:
                continue

            n_rows = len(features)
            # Pad or truncate
            if n_rows >= seq_len:
                features = features[:seq_len]
                mask = np.ones(seq_len, dtype=np.float32)
            else:
                pad_len = seq_len - n_rows
                features = np.vstack([
                    features,
                    np.zeros((pad_len, len(feature_cols)), dtype=np.float32),
                ])
                mask = np.concatenate([
                    np.ones(n_rows, dtype=np.float32),
                    np.zeros(pad_len, dtype=np.float32),
                ])

            self.sequences.append(features)
            self.targets.append(float(target))
            self.masks.append(mask)

        logger.info(
            f"Built {len(self.sequences)} route sequences "
            f"(seq_len={seq_len}, features={len(feature_cols)})"
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.sequences[idx]),   # (seq_len, n_features)
            torch.tensor(self.targets[idx]),          # scalar
            torch.from_numpy(self.masks[idx]),        # (seq_len,)
        )


def build_sequence_dataset(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COLUMN,
    seq_len: int = 10,
) -> RouteSequenceDataset:
    """Build a RouteSequenceDataset from a Polars DataFrame."""
    return RouteSequenceDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len,
    )
