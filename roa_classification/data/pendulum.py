"""Pendulum classification DataModule."""

from typing import Optional

from roa_classification.systems.pendulum import PendulumSystem

from .base import BaseClassificationDataModule


class PendulumClassificationDataModule(BaseClassificationDataModule):
    """DataModule for pendulum ROA classification.

    Embedded input: [sin(theta), cos(theta), theta_dot_norm] (3D).
    """

    def __init__(self, dataset_dir: str, **kwargs) -> None:
        super().__init__(data_dir=dataset_dir, **kwargs)
        self.dataset_dir_str = dataset_dir

    def setup(self, stage: Optional[str] = None) -> None:
        self.system = PendulumSystem(dataset_dir=self.dataset_dir_str)
        self.trajectories_dir = self.data_dir / "trajectories"
        super().setup(stage)
