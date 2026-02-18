"""Quadrotor 3D classification DataModule."""

from typing import Optional

from roa_classification.systems.quadrotor3d import Quadrotor3DSystem

from .base import BaseClassificationDataModule


class Quadrotor3DClassificationDataModule(BaseClassificationDataModule):
    """DataModule for Quadrotor 3D ROA classification.

    Train/val: 95/5 split of trajectory states (each embedded to 13D).
    Test: initial states from eval_states.txt.
    """

    def __init__(self, dataset_dir: str, **kwargs) -> None:
        super().__init__(data_dir=dataset_dir, **kwargs)
        self.dataset_dir_str = dataset_dir

    def setup(self, stage: Optional[str] = None) -> None:
        self.system = Quadrotor3DSystem(dataset_dir=self.dataset_dir_str)
        super().setup(stage)
