"""Shared data-loading logic for ROA classification."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import lightning as pl


class TrajectoryClassificationDataset(Dataset):
    """Load initial states from trajectory files and return embedded vectors.

    Each sample is the *first line* of a trajectory file paired with a 0/1 label.
    Subclass must supply an ``embed_state`` callable.
    """

    def __init__(
        self,
        trajectory_files: List[str],
        labels: List[int],
        trajectories_dir: Path,
        embed_fn,
    ) -> None:
        self.trajectory_files = trajectory_files
        self.labels = labels
        self.trajectories_dir = Path(trajectories_dir)
        self.embed_fn = embed_fn

    def __len__(self) -> int:
        return len(self.trajectory_files)

    def __getitem__(self, idx: int):
        filepath = self.trajectories_dir / self.trajectory_files[idx]
        with open(filepath) as f:
            first_line = f.readline().strip()
        state = np.array(
            [float(v) for v in first_line.split(",")], dtype=np.float32
        )
        embedded = self.embed_fn(state)
        return {
            "inputs": torch.from_numpy(embedded).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class BaseClassificationDataModule(pl.LightningDataModule):
    """Shared train/val splitting, balancing, and data-loader creation.

    Subclasses set ``self.system`` and ``self.trajectories_dir`` before
    calling ``super().setup()``.
    """

    def __init__(
        self,
        data_dir: str,
        split_index: int = 0,
        train_split: float = 0.95,
        batch_size: int = 256,
        num_workers: int = 4,
        max_samples: int = 0,
        balance_samples: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_index = split_index
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.balance_samples = balance_samples

        # Subclass must set these before calling super().setup()
        self.system = None
        self.trajectories_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        splits_dir = self.data_dir / "train_test_splits"
        idx_file = splits_dir / f"shuffled_indices_{self.split_index}.txt"
        lbl_file = splits_dir / f"shuffled_labels_{self.split_index}.txt"

        with open(idx_file) as f:
            self.trajectory_files = [line.strip() for line in f if line.strip()]
        with open(lbl_file) as f:
            self.labels = [int(line.strip()) for line in f if line.strip()]

        assert len(self.trajectory_files) == len(self.labels)

        if self.max_samples > 0:
            self.trajectory_files = self.trajectory_files[: self.max_samples]
            self.labels = self.labels[: self.max_samples]

    def setup(self, stage: Optional[str] = None) -> None:
        n = len(self.trajectory_files)
        train_end = int(n * self.train_split)

        train_files = self.trajectory_files[:train_end]
        train_labels = self.labels[:train_end]
        val_files = self.trajectory_files[train_end:]
        val_labels = self.labels[train_end:]

        embed_fn = self.system.embed_state

        self.train_dataset = TrajectoryClassificationDataset(
            train_files, train_labels, self.trajectories_dir, embed_fn
        )
        self.val_dataset = TrajectoryClassificationDataset(
            val_files, val_labels, self.trajectories_dir, embed_fn
        )

        # Compute sampler weights for class balancing
        if self.balance_samples:
            arr = np.array(train_labels)
            counts = np.bincount(arr, minlength=2).astype(float)
            weights = 1.0 / counts
            self._sample_weights = [weights[l] for l in train_labels]

    def train_dataloader(self) -> DataLoader:
        if self.balance_samples and hasattr(self, "_sample_weights"):
            sampler = WeightedRandomSampler(
                self._sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
