"""Unified training entry point.

Usage:
    python -m roa_classification.train system=cartpole
    python -m roa_classification.train system=pendulum model=mlp_large trainer.max_epochs=1000
"""

import hydra
import lightning as pl
import numpy as np
from omegaconf import DictConfig

BANNER = "=" * 80


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    system_name = cfg.system.name.replace("_", " ").title()
    print(f"\n{BANNER}")
    print(f"{system_name} Classification Training")
    print(BANNER)
    print(f"Config: {cfg.system.name}")
    print(f"Seed: {cfg.seed}")
    print(f"Max epochs: {cfg.trainer.max_epochs}")
    print(BANNER)

    # Data
    print("\nLoading data...")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    train_labels = data_module.train_labels.astype(int)
    n_train = len(data_module.train_dataset)
    n_val = len(data_module.val_dataset)
    n_test = len(data_module.test_dataset)
    counts = np.bincount(train_labels, minlength=2)
    n_traj = len(data_module.trajectory_files)

    print(f"Loaded {n_train + n_val} samples from {n_traj} trajectories")
    print(f"  Success (1): {counts[1]} ({counts[1] / n_train * 100:.1f}%)")
    print(f"  Failure (0): {counts[0]} ({counts[0] / n_train * 100:.1f}%)")
    print(f"Train: {n_train} samples (95%)")
    print(f"Val:   {n_val} samples (5%)")
    print(f"Test:  {n_test} samples (eval_states.txt)")

    # Model
    print("\nCreating model...")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ClassifierMLP")
    print(f"Input dim: {cfg.model.input_dim}")
    print(f"Hidden layers: {list(cfg.model.hidden_layers)}")
    print(f"Parameters: {n_params:,}")

    # Callbacks
    callbacks = [
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    ckpt_callback = None
    if "callbacks" in cfg:
        for _, cb_cfg in cfg.callbacks.items():
            cb = hydra.utils.instantiate(cb_cfg)
            callbacks.append(cb)
            if isinstance(cb, pl.pytorch.callbacks.ModelCheckpoint):
                ckpt_callback = cb

    # Logger
    logger = None
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)

    # Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    print(f"\nStarting training...")
    print(BANNER)

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print(f"\n{BANNER}")
    print("Training complete!")
    if ckpt_callback and ckpt_callback.best_model_path:
        print(f"Best checkpoint: {ckpt_callback.best_model_path}")
    print(BANNER)


if __name__ == "__main__":
    main()
