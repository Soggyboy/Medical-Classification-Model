"""
config.py — Centralized configuration for Dose Frequency Classifier.
Defines training, model, and preprocessing parameters.
"""

from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Data ---
    csv_path: str = "./data/doses.csv"   # default local path
    test_size: float = 0.2               # 80/20 split
    random_state: int = 42               # for reproducibility
    min_class_count: int = 2             # minimum samples per class to keep

    # --- TF-IDF Vectorizer ---
    ngram_min: int = 1
    ngram_max: int = 4
    max_features: int = 5000
    min_df: int = 2

    # --- Model ---
    hidden1: int = 512
    hidden2: int = 128
    num_classes: int = 343
    dropout: float = 0.0                 # adjustable later if needed

    # --- Training ---
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 10
    weight_decay: float = 0.0            # optional regularization
    patience: int = 10                   # early stopping patience

    # --- Device ---
    use_cuda: bool = True

    # --- Output ---
    save_dir: str = "./models"
    vec_path: str = "tfidf_vec.pkl"
    model_path: str = "flat343.pt"

    def device(self):
        """Return the appropriate torch.device string."""
        import torch
        return torch.device("cuda" if self.use_cuda and torch.cuda.is_available() else "cpu")
