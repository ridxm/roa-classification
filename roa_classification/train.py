"""Unified training entry point.

Usage:
    python -m roa_classification.train system=cartpole
    python -m roa_classification.train system=pendulum model=mlp_large trainer.max_epochs=1000
"""

import hydra
import lightning as pl
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate data module
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # Instantiate model
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    # Callbacks
    callbacks = []
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
