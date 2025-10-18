"""
encode.py — Target encoding utilities.

- create_multitask_target_tensor: builds (Primary, Secondary, Tertiary) integer tuples
  and returns the augmented DataFrame + label vocabulary.
- tuple_to_id / id_to_tuple: pack/unpack a 3-digit base-7 tuple into a single class id.
- make_texts_and_labels: convenience to produce model-ready arrays from the encoded df.
"""

from __future__ import annotations
from typing import Dict, Tuple, Iterable
import numpy as np
import pandas as pd

LabelVocab = Dict[str, int]


def create_multitask_target_tensor(df_in: pd.DataFrame) -> tuple[pd.DataFrame, LabelVocab]:
    """
    From a DataFrame with string columns:
        ['Primary class', 'Secondary class', 'Tertiary class']
    create integer-encoded tuples and return (augmented_df, label_to_int).

    Notes:
      - Reserves integer 0 for 'None'
      - All observed labels across the three columns share one vocabulary
    """
    df = df_in.copy()

    # Collect unique labels across the three columns
    all_labels = set(df["Primary class"].unique()) | \
                 set(df["Secondary class"].unique()) | \
                 set(df["Tertiary class"].unique())

    # Remove missing-like markers
    all_labels.discard(None)
    # np.nan may sneak in; discard safely
    if any(isinstance(x, float) and np.isnan(x) for x in all_labels):
        all_labels = {x for x in all_labels if not (isinstance(x, float) and np.isnan(x))}

    # Build vocabulary: 0 -> 'None', others start at 1
    label_to_int: LabelVocab = {label: i + 1 for i, label in enumerate(sorted(all_labels))}
    label_to_int["None"] = 0

    # Map each column
    primary_encoded = df["Primary class"].fillna("None").map(label_to_int)
    secondary_encoded = df["Secondary class"].fillna("None").map(label_to_int)
    tertiary_encoded = df["Tertiary class"].fillna("None").map(label_to_int)

    # Create tuple targets
    df["target_tuple"] = list(zip(primary_encoded, secondary_encoded, tertiary_encoded))

    return df, label_to_int


def tuple_to_id(t: tuple[int, int, int]) -> int:
    """
    Pack a 3-digit tuple (each digit in [0..6]) into a single class id in [0..342].
    id = 49*a + 7*b + c
    """
    a, b, c = int(t[0]), int(t[1]), int(t[2])
    return a * 49 + b * 7 + c


def id_to_tuple(i: int) -> tuple[int, int, int]:
    """
    Inverse of tuple_to_id.
    """
    a = i // 49
    b = (i % 49) // 7
    c = i % 7
    return (a, b, c)


def make_texts_and_labels(
    encoded_df: pd.DataFrame,
    text_col: str = "doses",
    tuple_col: str = "target_tuple",
    filter_singletons: bool = True,
    num_classes: int = 343,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Convenience helper: from an encoded DataFrame produce (texts, labels[, keep_mask])

    Args:
        encoded_df: must contain `text_col` and `tuple_col`.
        text_col: column with raw/normalized text strings.
        tuple_col: column containing 3-int tuples.
        filter_singletons: if True, drop classes with only 1 sample (required for
                           stratified splits).
        num_classes: total flat classes (should be 7*7*7 = 343).

    Returns:
        texts: 1D np.ndarray[str]
        labels: 1D np.ndarray[int] (packed ids)
        keep_mask: if filter_singletons=True, boolean mask applied to drop singletons;
                   otherwise None.
    """
    texts = encoded_df[text_col].astype(str).to_numpy()
    labels = np.array([tuple_to_id(t) for t in encoded_df[tuple_col]], dtype=np.int64)

    if not filter_singletons:
        return texts, labels, None

    counts = np.bincount(labels, minlength=num_classes)
    keep_mask = counts[labels] >= 2
    return texts[keep_mask], labels[keep_mask], keep_mask


# Optional: small sanity tests (run manually)
if __name__ == "__main__":
    # roundtrip check
    for a in range(7):
        for b in range(7):
            for c in range(7):
                idx = tuple_to_id((a, b, c))
                assert id_to_tuple(idx) == (a, b, c)
    print("encode.py roundtrip OK")
