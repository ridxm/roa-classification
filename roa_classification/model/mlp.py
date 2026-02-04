"""Binary classifier MLP as a Lightning module."""

from typing import List

import torch
import torch.nn as nn
import lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class ClassifierMLP(pl.LightningModule):
    """Simple feedforward MLP for binary ROA classification.

    Uses BCEWithLogitsLoss, AdamW optimizer, and CosineAnnealingLR scheduler.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        T_max: int = 500,
        eta_min: float = 1e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build MLP
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_cm = BinaryConfusionMatrix()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["label"])
        self.train_acc(logits, batch["label"].int())
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["label"])
        self.val_acc(logits, batch["label"].int())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["label"])
        labels_int = batch["label"].int()
        self.test_precision(logits, labels_int)
        self.test_recall(logits, labels_int)
        self.test_f1(logits, labels_int)
        self.test_cm.update(logits, labels_int)
        self.log("test/loss", loss)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        print(f"\nConfusion Matrix:\n{cm}")
        self.test_cm.reset()

    # ------------------------------------------------------------------
    # Optimizer / Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min,
        )
        return {"optimizer": optimizer, "scheduler": scheduler}
