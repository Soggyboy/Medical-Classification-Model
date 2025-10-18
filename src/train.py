"""
train.py — Training utilities for the Dose Frequency Classifier.

Features:
- Reproducible seeding
- Early stopping (val loss or accuracy)
- CosineAnnealingLR scheduler (optional)
- Inline live plotting for notebooks (optional)
- Checkpoint saving (best + last)

Usage example:
    from src.train import train
    history = train(model, train_loader, val_loader, device,
                    epochs=10, lr=1e-3, outdir="models",
                    class_weights=None, early_stopping_patience=10,
                    plot_live=True)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import os
import math
import numpy as np
import torch
import torch.nn as nn

# Optional plotting (only used when plot_live=True)
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


# --------------------------
# Reproducibility
# --------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Validation helper
# --------------------------

@torch.no_grad()
def validate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
             criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return {
        "loss": total_loss / max(n, 1),
        "acc":  correct / max(n, 1),
        "n":    n,
    }


# --------------------------
# One training epoch
# --------------------------

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return {
        "loss": total_loss / max(n, 1),
        "acc":  correct / max(n, 1),
        "n":    n,
    }


# --------------------------
# Early stopping
# --------------------------

@dataclass
class EarlyStopper:
    patience: int = 10
    mode: str = "min"  # "min" (val loss) or "max" (val acc)
    best: Optional[float] = None
    wait: int = 0
    stopped: bool = False

    def better(self, val: float) -> bool:
        if self.best is None:
            return True
        return (val < self.best) if self.mode == "min" else (val > self.best)

    def step(self, val: float) -> bool:
        if self.better(val):
            self.best = val
            self.wait = 0
            self.stopped = False
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped = True
        return self.stopped


# --------------------------
# Training loop (with optional live plotting)
# --------------------------

def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    scheduler: str = "cosine",          # "cosine" or "none"
    early_stopping_patience: int = 10,
    early_stopping_mode: str = "min",   # "min" on val_loss, "max" on val_acc
    outdir: str = "models",
    save_best: bool = True,
    save_last: bool = True,
    plot_live: bool = True,             # notebook-friendly live plot
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Train a model with validation, optional LR scheduler, early stopping, and live plot.

    Returns:
        history dict with lists: train_loss, val_loss, train_acc, val_acc
    """
    set_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # Criterion (optional class weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == "cosine":
        # Cosine to near-zero over total epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    else:
        sched = None

    # Early stopping monitor
    stopper = EarlyStopper(patience=early_stopping_patience, mode=early_stopping_mode)

    # Live plot setup
    if plot_live:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        (l_tr,), (l_va,) = ax[0].plot([], [], label="Train Loss"), ax[0].plot([], [], label="Val Loss")
        (a_tr,), (a_va,) = ax[1].plot([], [], label="Train Acc"),  ax[1].plot([], [], label="Val Acc")
        for a, title, ylab in zip(ax, ["Loss", "Accuracy"], ["Loss", "Accuracy"]):
            a.set_title(title); a.set_xlabel("Epoch"); a.set_ylabel(ylab); a.legend()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train / Validate
        train_out = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_out   = validate(model, val_loader, device, criterion)

        # Scheduler step (post-epoch)
        if sched is not None:
            sched.step()

        # Record
        history["train_loss"].append(train_out["loss"])
        history["train_acc"].append(train_out["acc"])
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(val_out["acc"])

        # Logging
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"lr {lr_now:.2e} | "
              f"train {train_out['loss']:.4f}/{train_out['acc']:.3f} | "
              f"val {val_out['loss']:.4f}/{val_out['acc']:.3f}")

        # Save best
        monitor_val = val_out["loss"] if stopper.mode == "min" else val_out["acc"]
        if save_best and (stopper.best is None or stopper.better(monitor_val)):
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            }
            torch.save(best_state, os.path.join(outdir, "best.ckpt"))

        # Live plot update
        if plot_live:
            xs = np.arange(1, epoch + 1)
            l_tr.set_data(xs, history["train_loss"]); l_va.set_data(xs, history["val_loss"])
            a_tr.set_data(xs, history["train_acc"]);  a_va.set_data(xs, history["val_acc"])
            for a in ax: a.relim(); a.autoscale_view()
            fig.tight_layout(); clear_output(wait=True); display(fig)

        # Early stopping
        if stopper.step(monitor_val):
            print(f"Early stopping at epoch {epoch} (best metric: {stopper.best:.4f})")
            break

    if plot_live:
        plt.close(fig)

    # Save last
    if save_last:
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            },
            os.path.join(outdir, "last.ckpt"),
        )

    # Also save best standalone weights for convenience
    if best_state is not None:
        torch.save(best_state["model"], os.path.join(outdir, "best_weights.pt"))

    return history
