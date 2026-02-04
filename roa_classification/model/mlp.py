"""Binary classifier MLP as a Lightning module."""

from typing import List

import torch
import torch.nn as nn
import lightning as pl
from torchmetrics.classification import BinaryAccuracy


class ClassifierMLP(pl.LightningModule):
    """Feedforward MLP for binary ROA classification.

    Uses BCEWithLogitsLoss, AdamW optimizer, and CosineAnnealingLR scheduler.
    Threshold-based evaluation: predictions below ``lower_thresh`` are class 0,
    above ``upper_thresh`` are class 1, and in-between is the separatrix
    (unclassified). Metrics are computed on classified samples only.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        T_max: int = 500,
        eta_min: float = 1e-6,
        lower_thresh: float = 0.4,
        upper_thresh: float = 0.6,
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
        self.best_val_loss = float("inf")

        # Buffers for threshold-based eval (collected across batches)
        self._val_probs: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []
        self._test_probs: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []

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
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["label"])
        self.val_acc(logits, batch["label"].int())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val_loss", loss)  # underscore variant for checkpoint filename
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self._val_probs.append(torch.sigmoid(logits).detach())
        self._val_labels.append(batch["label"].int().detach())

    def on_validation_epoch_end(self):
        if self._val_probs:
            probs = torch.cat(self._val_probs)
            labels = torch.cat(self._val_labels)
            # Track best val loss
            val_loss = self.trainer.callback_metrics.get("val/loss")
            if val_loss is not None and val_loss.item() < self.best_val_loss:
                self.best_val_loss = val_loss.item()
            self.log("val/loss_best", self.best_val_loss, prog_bar=True)
            self._compute_and_log_metrics(probs, labels, "val")
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        logits = self(batch["inputs"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("test/loss", loss)
        self._test_probs.append(torch.sigmoid(logits).detach())
        self._test_labels.append(batch["label"].int().detach())

    def on_test_epoch_end(self):
        if self._test_probs:
            probs = torch.cat(self._test_probs)
            labels = torch.cat(self._test_labels)
            self._compute_and_log_metrics(probs, labels, "test")
        self._test_probs.clear()
        self._test_labels.clear()

    # ------------------------------------------------------------------
    # Threshold-based metrics
    # ------------------------------------------------------------------

    def _compute_and_log_metrics(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> None:
        lo = self.hparams.lower_thresh
        hi = self.hparams.upper_thresh
        n = probs.numel()

        sep_mask = (probs >= lo) & (probs <= hi)
        classified_mask = ~sep_mask
        sep_pct = sep_mask.float().sum().item() / n * 100

        # Predictions for classified samples
        preds = (probs[classified_mask] >= hi).int()
        gt = labels[classified_mask]

        tp = ((preds == 1) & (gt == 1)).sum().float().item()
        fp = ((preds == 1) & (gt == 0)).sum().float().item()
        fn = ((preds == 0) & (gt == 1)).sum().float().item()
        tn = ((preds == 0) & (gt == 0)).sum().float().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        # Log all metrics
        self.log(f"{prefix}/acc", acc)
        self.log(f"{prefix}/precision", precision)
        self.log(f"{prefix}/recall", recall)
        self.log(f"{prefix}/specificity", specificity)
        self.log(f"{prefix}/f1", f1)
        self.log(f"{prefix}/separatrix_pct", sep_pct)
        self.log(f"{prefix}/tp", tp)
        self.log(f"{prefix}/fp", fp)
        self.log(f"{prefix}/tn", tn)
        self.log(f"{prefix}/fn", fn)
        self.log(f"{prefix}/tpr", tpr)
        self.log(f"{prefix}/fpr", fpr)
        self.log(f"{prefix}/tnr", tnr)
        self.log(f"{prefix}/fnr", fnr)

        # Console output
        n_cls = int(tp + fp + fn + tn)
        print(
            f"\n[{prefix}] thresholds=({lo}, {hi})  "
            f"total={n}  classified={n_cls}"
        )
        print(
            f"  {'Separatrix':>12s}  {'Precision':>10s}  {'Recall':>10s}  "
            f"{'Specificity':>12s}  {'F1':>10s}"
        )
        print(
            f"  {sep_pct:>11.2f}%  {precision:>10.4f}  {recall:>10.4f}  "
            f"{specificity:>12.4f}  {f1:>10.4f}"
        )

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
