"""
plots.py — Matplotlib plotting utilities.

Contains:
- plot_confusion_for_digit: 7x7 confusion matrix for one tuple digit (a/b/c)
- plot_confusion_grid: three side-by-side confusion matrices (a,b,c)
- plot_training_history: loss/accuracy curves from train() history dict

Notes:
- Works with arrays returned by evaluate(): true_tuples, pred_tuples.
- Set normalize=True to show row-normalized percentages instead of counts.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _annotate_cells(ax, mat: np.ndarray):
    vmax = mat.max() if mat.size else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.2f}" if isinstance(val, float) else f"{val}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if (val > 0.5 * vmax) else "black",
                fontsize=9,
            )


def _normalize_rows(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (cm / row_sums).round(2)


def plot_confusion_for_digit(
    true_tuples: np.ndarray,
    pred_tuples: np.ndarray,
    idx: int,
    *,
    normalize: bool = False,
    ax: Optional[plt.Axes] = None,
    title_prefix: str = "Tuple Digit",
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Plot a 7x7 confusion matrix for one tuple digit (idx in {0,1,2}).

    Returns: (fig, ax, cm) where cm is counts or row-normalized matrix.
    """
    assert idx in (0, 1, 2), "idx must be 0 (a), 1 (b), or 2 (c)"
    true_d = true_tuples[:, idx]
    pred_d = pred_tuples[:, idx]

    cm = confusion_matrix(true_d, pred_d, labels=range(7))
    cm_plot = _normalize_rows(cm) if normalize else cm

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm_plot, cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(7),
        yticks=np.arange(7),
        xticklabels=[f"{i}" for i in range(7)],
        yticklabels=[f"{i}" for i in range(7)],
        xlabel="Predicted",
        ylabel="True",
        title=f"{title_prefix} {idx} Confusion" + (" (norm)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    _annotate_cells(ax, cm_plot)
    fig.tight_layout()
    return fig, ax, cm


def plot_confusion_grid(
    true_tuples: np.ndarray,
    pred_tuples: np.ndarray,
    *,
    normalize: bool = False,
    figsize: Tuple[int, int] = (16, 4),
    titles: Tuple[str, str, str] = ("Digit a", "Digit b", "Digit c"),
) -> Tuple[plt.Figure, List[plt.Axes], List[np.ndarray]]:
    """
    Show three confusion matrices (a,b,c) side-by-side.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cms = []
    for d, ax, title in zip((0, 1, 2), axes, titles):
        f, a, cm = plot_confusion_for_digit(
            true_tuples, pred_tuples, d, normalize=normalize, ax=ax, title_prefix=title
        )
        cms.append(cm)
    fig.tight_layout()
    return fig, list(axes), cms


def plot_training_history(
    history: Dict[str, List[float]],
    *,
    figsize: Tuple[int, int] = (10, 4),
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot loss/accuracy curves from the history dict returned by train().
    Expected keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(epochs, history.get("train_loss", []), label="Train Loss")
    ax1.plot(epochs, history.get("val_loss", []), label="Val Loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()

    ax2.plot(epochs, history.get("train_acc", []), label="Train Acc")
    ax2.plot(epochs, history.get("val_acc", []), label="Val Acc")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)
