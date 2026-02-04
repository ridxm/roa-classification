"""CartPole PyBullet classification DataModule."""

from typing import Optional

from roa_classification.systems.cartpole import CartPoleSystem

from .base import BaseClassificationDataModule


class CartPoleClassificationDataModule(BaseClassificationDataModule):
    """DataModule for CartPole ROA classification.

    Embedded input:
        [x_norm, sin(theta), cos(theta), x_dot_norm, theta_dot_norm] (5D).
    """

    def __init__(self, dataset_dir: str, **kwargs) -> None:
        super().__init__(data_dir=dataset_dir, **kwargs)
        self.dataset_dir_str = dataset_dir

    def setup(self, stage: Optional[str] = None) -> None:
        self.system = CartPoleSystem(dataset_dir=self.dataset_dir_str)
        self.trajectories_dir = self.data_dir / "trajectories"
        super().setup(stage)
