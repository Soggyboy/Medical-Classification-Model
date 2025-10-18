"""
io_utils.py — Data input/output utilities for the Dose Frequency Classifier.

Handles:
- Safe CSV loading (local or Colab)
- Automatic cleaning of unwanted columns
- Optional uploading from Google Colab
"""

import os
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file with automatic delimiter detection and cleaning.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for processing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ CSV file not found at: {path}")

    # sep=None makes pandas detect the correct delimiter automatically
    df = pd.read_csv(path, sep=None, engine="python")

    # Remove redundant index columns if present
    redundant_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    if redundant_cols:
        df = df.drop(columns=redundant_cols)
        print(f"🧹 Dropped redundant column(s): {redundant_cols}")

    print(f"✅ Loaded CSV: {path} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def upload_csv_colab() -> pd.DataFrame:
    """
    Upload a CSV interactively in Google Colab.

    Returns:
        pd.DataFrame: Loaded and cleaned dataframe.
    """
    try:
        from google.colab import files
    except ImportError:
        raise EnvironmentError("This function is only available in Google Colab.")

    print("📤 Please upload your CSV file (e.g., 'dose_vbm.csv'):")
    uploaded = files.upload()

    # Get the first uploaded filename
    if not uploaded:
        raise RuntimeError("No file was uploaded.")

    filename = list(uploaded.keys())[0]
    print(f"✅ Received: {filename}")
    return load_csv(filename)
