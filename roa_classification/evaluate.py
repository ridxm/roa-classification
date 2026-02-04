"""Unified evaluation entry point.

Usage:
    python -m roa_classification.evaluate system=cartpole checkpoint=/path/to/ckpt
"""

import hydra
import lightning as pl
from omegaconf import DictConfig

from roa_classification.model.mlp import ClassifierMLP


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup(stage="test")

    # Load model from checkpoint
    model = ClassifierMLP.load_from_checkpoint(cfg.checkpoint)

    # Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
