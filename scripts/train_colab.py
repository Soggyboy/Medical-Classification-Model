# scripts/train_colab.py
import os
import pickle
import numpy as np
import pandas as pd
import torch
from google.colab import files
from sklearn.model_selection import StratifiedShuffleSplit

from src.io_utils import load_csv
from src.rules import classify_all
from src.encode import create_multitask_target_tensor, tuple_to_id
from src.vectorize import CharTfidf
from src.dataset import make_loaders
from src.model import FlatClassifier
from src.train import train
from src.evaluate import evaluate

# --------------------------
# Config (adjust as needed)
# --------------------------
EPOCHS = 10
BATCH_SIZE = 256
TEST_SIZE = 0.2
RANDOM_STATE = 42
NGRAM_RANGE = (1, 4)
MAX_FEATURES = 5000
MIN_DF = 2
OUTDIR = "/content/models"
TEXT_COL = "dose_vbm_org"  # column in your CSV containing the raw dose text

# --------------------------
# Upload CSV
# --------------------------
print("📤 Upload your CSV (must include a column named 'dose_vbm_org')...")
uploaded = files.upload()
assert uploaded, "No file uploaded."
csv_path = next(iter(uploaded.keys()))
print(f"✅ Received: {csv_path}")

# --------------------------
# Load & prepare data
# --------------------------
raw_df = load_csv(csv_path)  # auto-drops 'Unnamed: 0' if present

# Build rule-based labels (Primary/Secondary/Tertiary) and a working DataFrame
N = len(raw_df)
work = {"doses": [], "Primary class": [], "Secondary class": [], "Tertiary class": []}
for i in range(N):
    dose_value = raw_df.loc[i, TEXT_COL]
    cls = classify_all(dose_value)
    work["doses"].append(dose_value)
    work["Primary class"].append(cls[0] if len(cls) > 0 else None)
    work["Secondary class"].append(cls[1] if len(cls) > 1 else None)
    work["Tertiary class"].append(cls[2] if len(cls) > 2 else None)
df = pd.DataFrame.from_dict(work)

# Encode multitask targets and pack tuple -> single id
encoded_df, vocab = create_multitask_target_tensor(df)
texts = encoded_df["doses"].astype(str).to_numpy()
labels = np.array([tuple_to_id(t) for t in encoded_df["target_tuple"]], dtype=np.int64)

# Filter singleton classes (needed for stratified split)
num_classes = 343
counts = np.bincount(labels, minlength=num_classes)
keep_mask = counts[labels] >= 2
texts, labels = texts[keep_mask], labels[keep_mask]

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, val_idx = next(sss.split(texts, labels))
texts_train, texts_val = texts[train_idx], texts[val_idx]
y_train, y_val = labels[train_idx], labels[val_idx]

# Vectorize (char TF-IDF)
vec = CharTfidf(ngram_range=NGRAM_RANGE, max_features=MAX_FEATURES, min_df=MIN_DF)
X_train = vec.fit_transform(texts_train)
X_val   = vec.transform(texts_val)

# Dataloaders
train_loader, val_loader = make_loaders(
    X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE
)

# Model + device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlatClassifier(in_dim=X_train.shape[1]).to(device)

# --------------------------
# Train (with live plot in Colab)
# --------------------------
history = train(
    model, train_loader, val_loader, device,
    epochs=EPOCHS, lr=1e-3, weight_decay=0.0,
    scheduler="cosine",
    early_stopping_patience=10,
    early_stopping_mode="min",
    outdir=OUTDIR,
    save_best=True, save_last=True,
    plot_live=True,  # inline live plot in Colab
    seed=42,
)

# --------------------------
# Save artifacts
# --------------------------
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "tfidf_vec.pkl"), "wb") as f:
    pickle.dump(vec.vec, f)  # save underlying sklearn vectorizer
torch.save(model.state_dict(), os.path.join(OUTDIR, "flat343.pt"))
print(f"💾 Saved model + vectorizer to: {OUTDIR}")

# --------------------------
# Quick evaluation on the held-out split
# --------------------------
out = evaluate(model, val_loader, device)

print("✅ Done.")
