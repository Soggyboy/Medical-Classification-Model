"""
dataset.py — Torch dataset & dataloader utilities.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class TextDataset(Dataset):
    """
    Minimal dataset wrapping precomputed feature tensors and int labels.
    """
    def __init__(self, X_t: torch.Tensor, y_t: torch.Tensor):
        assert X_t.shape[0] == y_t.shape[0], "X and y must have same length"
        self.X_t = X_t
        self.y_t = y_t

    def __len__(self) -> int:
        return self.X_t.size(0)

    def __getitem__(self, i: int):
        return self.X_t[i], self.y_t[i]


def _seed_worker(worker_id: int):
    """
    Ensures dataloader workers are deterministically seeded (use with generator).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_tensordataset: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders from NumPy arrays produced by your vectorizer.

    Args:
        X_train, y_train, X_val, y_val: NumPy arrays.
        batch_size: loader batch size.
        num_workers: dataloader workers (0 is fine cross-platform).
        pin_memory: set True if using CUDA for small speedup.
        use_tensordataset: if True, use TensorDataset instead of custom class.

    Returns:
        (train_loader, val_loader)
    """
    # -> tensors
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val,   dtype=torch.float32)
    yva = torch.tensor(y_val,   dtype=torch.long)

    if use_tensordataset:
        train_ds = TensorDataset(Xtr, ytr)
        val_ds   = TensorDataset(Xva, yva)
    else:
        train_ds = TextDataset(Xtr, ytr)
        val_ds   = TextDataset(Xva, yva)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def class_weights(
    y: np.ndarray,
    num_classes: int,
    smoothing: float = 0.0,
    to_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced CrossEntropy.

    Args:
        y: 1D array of integer class ids.
        num_classes: total number of classes (e.g., 343).
        smoothing: label smoothing on counts (add to each class).
        to_device: optional torch.device to move the weights.

    Returns:
        torch.FloatTensor of shape [num_classes]
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts += float(smoothing)
    counts[counts == 0] = 1.0  # avoid div/0
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes  # normalize mean weight ~1
    w_t = torch.tensor(w, dtype=torch.float32)
    if to_device is not None:
        w_t = w_t.to(to_device)
    return w_t
