"""
infer.py — Inference utilities for the Dose Frequency Classifier.

- load_artifacts: load fitted vectorizer + model weights
- predict_texts: run top-k predictions on raw strings
- format_predictions: convert outputs to a tidy DataFrame
- main (CLI): quick inference from --text or a newline-delimited file

Artifacts:
- Vectorizer: saved via pickle (either sklearn TfidfVectorizer, or our CharTfidf/HybridTfidf wrapper)
- Weights:   torch saved state_dict (e.g., models/flat343.pt)

Example:
    from src.infer import load_artifacts, predict_texts, format_predictions
    vec, model, device = load_artifacts("models/tfidf_vec.pkl", "models/flat343.pt")
    outs = predict_texts(["take 1 tab bid", "2 qhs prn"], vec, model, device, topk=5)
    df = format_predictions(outs); print(df)
"""

from __future__ import annotations
from typing import List, Dict, Any, Iterable, Tuple
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .model import FlatClassifier, create_model
from .encode import id_to_tuple


# ---------------------------
# Artifacts loading
# ---------------------------

def _feature_dim_from_vec(vec) -> int:
    """
    Try to infer the number of output features from various vectorizer types.
    Works with:
      - sklearn TfidfVectorizer
      - our CharTfidf / HybridTfidf wrappers (vec.vec or combined)
    """
    # Our wrappers expose .vec (sklearn) or both .char_vec/.word_vec for Hybrid
    if hasattr(vec, "get_feature_names_out"):
        return len(vec.get_feature_names_out())

    # CharTfidf wrapper
    if hasattr(vec, "vec") and hasattr(vec.vec, "get_feature_names_out"):
        return len(vec.vec.get_feature_names_out())

    # HybridTfidf wrapper (char + word)
    if hasattr(vec, "char_vec") and hasattr(vec, "word_vec"):
        n_char = len(vec.char_vec.get_feature_names_out())
        n_word = len(vec.word_vec.get_feature_names_out())
        return n_char + n_word

    # Fallback: try transforming a tiny probe
    X = vec.transform(["probe"]).toarray()
    return X.shape[1]


def load_artifacts(vec_path: str, weights_path: str, in_dim: int | None = None,
                   device: torch.device | None = None) -> Tuple[Any, torch.nn.Module, torch.device]:
    """
    Load vectorizer + model weights. If in_dim is None, infer from vectorizer.

    Args:
        vec_path: pickle path to fitted vectorizer
        weights_path: torch weights (state_dict) path
        in_dim: optional explicit input dim override
        device: optional torch.device override

    Returns:
        (vectorizer, model (eval mode), device)
    """
    # Load vectorizer (sklearn or our wrapper)
    with open(vec_path, "rb") as f:
        vec = pickle.load(f)

    # Infer device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer input dimension if not supplied
    if in_dim is None:
        in_dim = _feature_dim_from_vec(vec)

    # Build model & load weights
    model = FlatClassifier(in_dim=in_dim)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return vec, model, device


# ---------------------------
# Prediction
# ---------------------------

@torch.no_grad()
def predict_texts(raw_texts: Iterable[str], vec, model: torch.nn.Module, device: torch.device,
                  topk: int = 3) -> List[Dict[str, Any]]:
    """
    Run inference on a batch of strings and return top-k candidates.

    Returns a list of:
        {
          "text": str,
          "pred_id": int,
          "pred_tuple": (a,b,c),
          "topk": List[(id, tuple, prob)]
        }
    """
    # Vectorize
    # Support both sklearn vectorizers and our wrapper classes
    if hasattr(vec, "transform"):
        # sklearn or Hybrid (returns sparse)
        X = vec.transform(np.array(list(raw_texts), dtype=str))
        # Convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()
    else:
        # Unlikely, but just in case someone passes raw callable
        X = vec(np.array(list(raw_texts), dtype=str))

    X = X.astype("float32")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    # Forward
    logits = model(X_t)
    probs = F.softmax(logits, dim=1).cpu().numpy()

    # Top-k
    topk_ids = np.argsort(-probs, axis=1)[:, :topk]
    topk_probs = np.take_along_axis(probs, topk_ids, axis=1)

    preds: List[Dict[str, Any]] = []
    texts = list(raw_texts)
    for i, text in enumerate(texts):
        best_id = int(topk_ids[i, 0])
        best_tuple = id_to_tuple(best_id)
        candidates = [
            (int(topk_ids[i, j]), id_to_tuple(int(topk_ids[i, j])), float(topk_probs[i, j]))
            for j in range(topk)
        ]
        preds.append(
            {
                "text": text,
                "pred_id": best_id,
                "pred_tuple": best_tuple,
                "topk": candidates,
            }
        )
    return preds


def format_predictions(preds: List[Dict[str, Any]], topk: int | None = None) -> pd.DataFrame:
    """
    Turn prediction dicts into a tidy DataFrame.
    If topk is provided, include those many candidate columns.
    """
    rows = []
    for p in preds:
        row = {
            "text": p["text"],
            "pred_id": p["pred_id"],
            "pred_tuple": p["pred_tuple"],
        }
        if topk is None and "topk" in p:
            topk = len(p["topk"])
        if topk:
            for j in range(min(topk, len(p.get("topk", [])))):
                cid, tup, prob = p["topk"][j]
                row[f"top{j+1}_id"] = cid
                row[f"top{j+1}_tuple"] = tup
                row[f"top{j+1}_prob"] = prob
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------
# CLI (optional)
# ---------------------------

def _read_texts_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    """
    Minimal CLI:
      python -m src.infer --vec models/tfidf_vec.pkl --ckpt models/flat343.pt --text "1 tab bid" "2 qhs prn"
      python -m src.infer --vec models/tfidf_vec.pkl --ckpt models/flat343.pt --file samples.txt --topk 5
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vec", required=True, help="Path to fitted vectorizer pickle")
    parser.add_argument("--ckpt", required=True, help="Path to model state_dict (.pt)")
    parser.add_argument("--text", nargs="*", help="Raw dose strings")
    parser.add_argument("--file", help="Path to newline-delimited text file")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide --text items or --file path")

    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        texts.extend(_read_texts_from_file(args.file))

    vec, model, device = load_artifacts(args.vec, args.ckpt)
    outs = predict_texts(texts, vec, model, device, topk=args.topk)
    df = format_predictions(outs, topk=args.topk)
    # Pretty print
    with pd.option_context("display.max_colwidth", 120):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
