"""
evaluate.py — Validation metrics and reporting.

- evaluate: runs the model on a validation loader and returns:
    - metrics_df: 1-row summary (loss/acc/top3/top5)
    - classification_df: sklearn classification_report as a DataFrame
    - y_true, y_pred, logits
    - true_tuples, pred_tuples (decoded via id_to_tuple)
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

# Optional pretty display in notebooks
try:
    from IPython.display import display
    _HAS_DISPLAY = True
except Exception:
    _HAS_DISPLAY = False

from .encode import id_to_tuple


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             device: torch.device) -> Dict[str, Any]:
    """
    Evaluate model on a validation DataLoader.

    Args:
        model: trained torch.nn.Module
        val_loader: DataLoader yielding (features, labels)
        device: torch.device

    Returns:
        dict with:
            - metrics_df: pd.DataFrame with Loss, Accuracy, Top-3 Accuracy, Top-5 Accuracy
            - classification_df: pd.DataFrame with sklearn's classification_report
            - y_true, y_pred: np.ndarray
            - logits: np.ndarray (float) of raw scores
            - true_tuples, pred_tuples: np.ndarray of shape [N, 3]
    """
    model.eval()
    crit = torch.nn.CrossEntropyLoss(reduction="sum")
    all_logits, all_y = [], []
    total_loss, n = 0.0, 0

    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        total_loss += loss.item()
        n += xb.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(yb.detach().cpu())

    logits = torch.cat(all_logits, 0).numpy()
    y_true = torch.cat(all_y, 0).numpy()
    y_pred = logits.argmax(1)

    # Summary metrics
    val_loss = float(total_loss) / max(int(n), 1)
    acc = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, logits, k=3, labels=np.arange(logits.shape[1]))
    top5 = top_k_accuracy_score(y_true, logits, k=5, labels=np.arange(logits.shape[1]))

    metrics_df = pd.DataFrame([{
        "Loss": val_loss,
        "Accuracy": acc,
        "Top-3 Accuracy": top3,
        "Top-5 Accuracy": top5,
    }])

    # Detailed per-class report
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    classification_df = pd.DataFrame(report_dict).transpose()

    # Decode integer ids → tuple digits (a,b,c)
    true_tuples = np.array([id_to_tuple(int(i)) for i in y_true])
    pred_tuples = np.array([id_to_tuple(int(i)) for i in y_pred])

    # Nice display in notebooks; fallback to print otherwise
    if _HAS_DISPLAY:
        print("📊 Overall Validation Metrics:")
        display(metrics_df)
        print("\n📋 Detailed Per-Class Report:")
        display(classification_df)
    else:
        # Headless/logging environments
        print("Overall Validation Metrics:\n", metrics_df.to_string(index=False))
        print("\nDetailed Per-Class Report:\n", classification_df.to_string())

    return {
        "metrics_df": metrics_df,
        "classification_df": classification_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits,
        "true_tuples": true_tuples,
        "pred_tuples": pred_tuples,
    }
