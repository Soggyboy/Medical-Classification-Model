# scripts/eval.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.io_utils import load_csv
from src.rules import classify_all
from src.encode import create_multitask_target_tensor, tuple_to_id
from src.model import FlatClassifier
from src.evaluate import evaluate
from src.plots import plot_confusion_for_digit


def build_training_dataframe(raw_df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run rule-based extraction to build (doses, Primary/Secondary/Tertiary) columns."""
    N = len(raw_df)
    rows = {"doses": [], "Primary class": [], "Secondary class": [], "Tertiary class": []}
    for i in range(N):
        dose_value = raw_df.loc[i, text_col]
        cls = classify_all(dose_value)
        rows["doses"].append(dose_value)
        rows["Primary class"].append(cls[0] if len(cls) > 0 else None)
        rows["Secondary class"].append(cls[1] if len(cls) > 1 else None)
        rows["Tertiary class"].append(cls[2] if len(cls) > 2 else None)
    return pd.DataFrame.from_dict(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Dose Frequency Classifier.")
    parser.add_argument("--csv", required=True, help="Path to CSV containing the text column.")
    parser.add_argument("--text_col", default="dose_vbm_org", help="Name of the dose text column.")
    parser.add_argument("--ckpt", required=True, help="Path to model weights (state_dict .pt).")
    parser.add_argument("--vec", required=True, help="Path to fitted TF-IDF vectorizer pickle.")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--outdir", default="models", help="Where to save evaluation artifacts.")
    parser.add_argument("--no_plots", action="store_true", help="Disable confusion-matrix plots.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data
    raw_df = load_csv(args.csv)
    if args.text_col not in raw_df.columns:
        raise KeyError(f"Column '{args.text_col}' not found in {args.csv}")

    # 2) Build rule-based labels and encode tuples
    training_df = build_training_dataframe(raw_df, args.text_col)
    encoded_df, _ = create_multitask_target_tensor(training_df)

    # 3) Pack tuple -> id and filter singletons (required for stratified split)
    texts = encoded_df["doses"].astype(str).to_numpy()
    labels = np.array([tuple_to_id(t) for t in encoded_df["target_tuple"]], dtype=np.int64)
    num_classes = 343
    counts = np.bincount(labels, minlength=num_classes)
    keep_mask = counts[labels] >= 2
    texts, labels = texts[keep_mask], labels[keep_mask]

    # 4) Split (stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, val_idx = next(sss.split(texts, labels))
    texts_val, y_val = texts[val_idx], labels[val_idx]

    # 5) Load vectorizer + transform val
    with open(args.vec, "rb") as f:
        vec = pickle.load(f)  # saved sklearn TfidfVectorizer
    X_val = vec.transform(texts_val).toarray().astype("float32")

    # 6) Model + device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlatClassifier(in_dim=X_val.shape[1]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # 7) Val DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # 8) Evaluate
    out = evaluate(model, val_loader, device)

    # 9) Save metrics
    out["metrics_df"].to_csv(os.path.join(args.outdir, "metrics_eval.csv"), index=False)
    out["classification_df"].to_csv(os.path.join(args.outdir, "classification_report_eval.csv"))
    print(f"Saved metrics to {args.outdir}")

    # 10) Plots (per-digit confusions)
    if not args.no_plots:
        for d in range(3):
            fig, _, _ = plot_confusion_for_digit(out["true_tuples"], out["pred_tuples"], idx=d, normalize=False)
            fig.savefig(os.path.join(args.outdir, f"confusion_digit_{d}.png"), dpi=160)
        print("Saved confusion matrices.")


if __name__ == "__main__":
    main()
