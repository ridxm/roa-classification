"""Unified training entry point.

Usage:
    python -m roa_classification.train system=cartpole
    python -m roa_classification.train system=pendulum model=mlp_large trainer.max_epochs=1000
"""

import logging

import hydra
import lightning as pl
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Instantiate data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # Log dataset stats
    train_labels = data_module.train_dataset.labels.astype(int)
    counts = np.bincount(train_labels, minlength=2)
    log.info(
        "Train: %d samples from %d trajectories (label 0: %d, label 1: %d)",
        len(data_module.train_dataset),
        len(data_module.trajectory_files),
        counts[0],
        counts[1],
    )
    log.info("Val: %d samples (eval_states)", len(data_module.val_dataset))

    # Instantiate model
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # Callbacks
    callbacks = [
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    if "callbacks" in cfg:
        for _, cb_cfg in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_cfg))

    # Logger
    logger = None
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)

    # Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
