# scripts/train_local.py
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
from src.vectorize import CharTfidf
from src.dataset import make_loaders, class_weights
from src.model import FlatClassifier, create_model
from src.train import train
from src.evaluate import evaluate


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
    parser = argparse.ArgumentParser(description="Train Dose Frequency Classifier locally.")
    parser.add_argument("--csv", required=True, help="Path to input CSV (must contain text column).")
    parser.add_argument("--text_col", default="dose_vbm_org", help="Name of the text column in CSV.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=4)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_class_weights", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--no_plot", action="store_true", help="Disable live plotting (headless).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load CSV
    raw_df = load_csv(args.csv)
    if args.text_col not in raw_df.columns:
        raise KeyError(f"Column '{args.text_col}' not found in {args.csv}")

    # 2) Build rule-based labels and encode tuples
    training_df = build_training_dataframe(raw_df, args.text_col)
    encoded_df, vocab = create_multitask_target_tensor(training_df)

    # 3) Pack tuple -> id and filter singleton classes (required for stratified split)
    texts = encoded_df["doses"].astype(str).to_numpy()
    labels = np.array([tuple_to_id(t) for t in encoded_df["target_tuple"]], dtype=np.int64)

    num_classes = 343
    counts = np.bincount(labels, minlength=num_classes)
    keep_mask = counts[labels] >= 2
    texts, labels = texts[keep_mask], labels[keep_mask]

    # 4) Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, val_idx = next(sss.split(texts, labels))
    texts_train, texts_val = texts[train_idx], texts[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # 5) Vectorize (char TF-IDF)
    vec = CharTfidf(ngram_range=(args.ngram_min, args.ngram_max),
                    max_features=args.max_features,
                    min_df=args.min_df)
    X_train = vec.fit_transform(texts_train)
    X_val   = vec.transform(texts_val)

    # 6) DataLoaders
    train_loader, val_loader = make_loaders(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )

    # 7) Model + device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlatClassifier(
        in_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)

    # Optional class weights
    weights = class_weights(y_train, num_classes, to_device=device) if args.use_class_weights else None

    # 8) Train (with optional live plot)
    history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        class_weights=weights,
        scheduler="cosine",
        early_stopping_patience=10,
        early_stopping_mode="min",
        outdir=args.outdir,
        save_best=True, save_last=True,
        plot_live=not args.no_plot,
        seed=args.random_state,
    )

    # 9) Save artifacts
    with open(os.path.join(args.outdir, "tfidf_vec.pkl"), "wb") as f:
        # save underlying sklearn vectorizer for portability
        pickle.dump(vec.vec, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "flat343.pt"))
    print(f"💾 Saved model + vectorizer to: {args.outdir}")

    # 10) Quick evaluation on the held-out split
    out = evaluate(model, val_loader, device)
    # Persist metrics for CI/repro
    out["metrics_df"].to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)
    out["classification_df"].to_csv(os.path.join(args.outdir, "classification_report.csv"))

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
