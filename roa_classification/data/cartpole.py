"""CartPole PyBullet classification DataModule."""

from typing import Optional

from roa_classification.systems.cartpole import CartPoleSystem

from .base import BaseClassificationDataModule


class CartPoleClassificationDataModule(BaseClassificationDataModule):
    """DataModule for CartPole ROA classification.

    Train/val: 95/5 split of trajectory states (each embedded to 5D).
    Test: initial states from eval_states.txt.
    """

    def __init__(self, dataset_dir: str, **kwargs) -> None:
        super().__init__(data_dir=dataset_dir, **kwargs)
        self.dataset_dir_str = dataset_dir

    def setup(self, stage: Optional[str] = None) -> None:
        self.system = CartPoleSystem(dataset_dir=self.dataset_dir_str)
        super().setup(stage)
