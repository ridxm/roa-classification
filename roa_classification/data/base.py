"""Shared data-loading logic for ROA classification."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import lightning as pl


class TrajectoryDataset(Dataset):
    """All states from trajectory files, pre-loaded into memory.

    Every state in a trajectory inherits that trajectory's 0/1 label.
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
    """Dataset loaded from eval_states.txt (used for final test).

    Format: initial_state..., final_state..., label
    Uses only the initial state (first ``state_dim`` cols) and label (last col).
    """

    def __init__(self, eval_states_path: Path, embed_fn, state_dim: int) -> None:
        states = []
        labels = []
        with open(eval_states_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = line.split(",")
                raw = np.array(
                    [float(v) for v in vals[:state_dim]], dtype=np.float32
                )
                states.append(embed_fn(raw))
                labels.append(int(float(vals[-1])))

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
    """Train/val from trajectory files (trajectory-level split), test on test_set.txt.

    Subclasses set ``self.system`` before calling ``super().setup()``.
    """

    def __init__(
        self,
        data_dir: str,
        split_index: int = 0,
        split_name: Optional[str] = None,
        num_trajectories: int = 1000,
        train_split: float = 0.9,
        batch_size: int = 256,
        num_workers: int = 4,
        balance_samples: bool = True,
        test_file: str = "test_set.txt",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_index = split_index
        self.split_name = split_name
        self.num_trajectories = num_trajectories
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_samples = balance_samples
        self.test_file = test_file

        self.system = None  # subclass sets this

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        splits_dir = self.data_dir / "train_test_splits"
        if self.split_name:
            idx_file = splits_dir / f"{self.split_name}_indices.txt"
            lbl_file = splits_dir / f"{self.split_name}_labels.txt"
        else:
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

        # Split trajectories for train/val (trajectory-level split)
        n_traj = len(self.trajectory_files)
        train_end = int(n_traj * self.train_split)

        self.train_trajectory_files = self.trajectory_files[:train_end]
        self.train_trajectory_labels = self.trajectory_labels[:train_end]
        self.val_trajectory_files = self.trajectory_files[train_end:]
        self.val_trajectory_labels = self.trajectory_labels[train_end:]

        # Create separate datasets for train and val
        self.train_dataset = TrajectoryDataset(
            self.train_trajectory_files,
            self.train_trajectory_labels,
            trajectories_dir,
            embed_fn,
        )
        self.val_dataset = TrajectoryDataset(
            self.val_trajectory_files,
            self.val_trajectory_labels,
            trajectories_dir,
            embed_fn,
        )

        # Store labels for logging
        self.train_labels = self.train_dataset.labels
        self.val_labels = self.val_dataset.labels

        # Test dataset
        self.test_dataset = EvalStatesDataset(
            self.data_dir / self.test_file, embed_fn, self.system.state_dim
        )

        # Compute sampler weights for class balancing on train split
        if self.balance_samples:
            counts = np.bincount(
                self.train_labels.astype(int), minlength=2
            ).astype(float)
            weights = 1.0 / counts
            self._sample_weights = [
                weights[int(l)] for l in self.train_labels
            ]

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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
