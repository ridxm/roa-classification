"""Shared data-loading logic for ROA classification."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import lightning as pl


class TrajectoryTrainDataset(Dataset):
    """Training dataset: all states from trajectory files.

    Every state in a trajectory inherits that trajectory's 0/1 label.
    States and labels are pre-loaded into memory as flat arrays.
    """

    def __init__(
        self,
        trajectory_files: List[str],
        labels: List[int],
        trajectories_dir: Path,
        embed_fn,
    ) -> None:
        all_states = []
        all_labels = []
        trajectories_dir = Path(trajectories_dir)

        for fname, label in zip(trajectory_files, labels):
            filepath = trajectories_dir / fname
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    state = np.array(
                        [float(v) for v in line.split(",")], dtype=np.float32
                    )
                    all_states.append(embed_fn(state))
                    all_labels.append(label)

        self.states = np.stack(all_states)
        self.labels = np.array(all_labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "inputs": torch.from_numpy(self.states[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class EvalStatesDataset(Dataset):
    """Validation dataset loaded from eval_states.txt.

    Each row: x,theta,x_dot,theta_dot,x_f,theta_f,x_dot_f,theta_dot_f,label
    Uses only the initial state (cols 0-3) and label (col 8).
    """

    def __init__(self, eval_states_path: Path, embed_fn) -> None:
        states = []
        labels = []
        with open(eval_states_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = line.split(",")
                raw = np.array([float(v) for v in vals[:4]], dtype=np.float32)
                states.append(embed_fn(raw))
                labels.append(int(float(vals[8])))

        self.states = np.stack(states)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "inputs": torch.from_numpy(self.states[idx]).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class BaseClassificationDataModule(pl.LightningDataModule):
    """Train on all states from trajectory files, validate on eval_states.txt.

    Subclasses set ``self.system`` before calling ``super().setup()``.
    """

    def __init__(
        self,
        data_dir: str,
        split_index: int = 0,
        num_trajectories: int = 1000,
        batch_size: int = 256,
        num_workers: int = 4,
        balance_samples: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_index = split_index
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_samples = balance_samples

        self.system = None  # subclass sets this

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        splits_dir = self.data_dir / "train_test_splits"
        idx_file = splits_dir / f"shuffled_indices_{self.split_index}.txt"
        lbl_file = splits_dir / f"shuffled_labels_{self.split_index}.txt"

        with open(idx_file) as f:
            all_files = [line.strip() for line in f if line.strip()]
        with open(lbl_file) as f:
            all_labels = [int(line.strip()) for line in f if line.strip()]

        assert len(all_files) == len(all_labels)

        n = min(self.num_trajectories, len(all_files))
        self.trajectory_files = all_files[:n]
        self.trajectory_labels = all_labels[:n]

    def setup(self, stage: Optional[str] = None) -> None:
        embed_fn = self.system.embed_state
        trajectories_dir = self.data_dir / "trajectories"

        self.train_dataset = TrajectoryTrainDataset(
            self.trajectory_files,
            self.trajectory_labels,
            trajectories_dir,
            embed_fn,
        )

        self.val_dataset = EvalStatesDataset(
            self.data_dir / "eval_states.txt", embed_fn
        )

        # Compute sampler weights for class balancing
        if self.balance_samples:
            labels = self.train_dataset.labels
            counts = np.bincount(labels.astype(int), minlength=2).astype(float)
            weights = 1.0 / counts
            self._sample_weights = [weights[int(l)] for l in labels]

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
