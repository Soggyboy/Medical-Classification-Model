"""
model.py — Neural architectures for the Dose Frequency Classifier.

Currently provides:
- FlatClassifier: MLP over TF-IDF features → 343-way softmax logits
- create_model: convenience factory from a TrainConfig
- count_parameters: utility to report trainable parameter count
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class FlatClassifier(nn.Module):
    """
    Simple feed-forward network for multi-class classification.

    Args:
        in_dim: input feature dimension (TF-IDF)
        num_classes: number of output classes (default 343)
        hidden1, hidden2: hidden layer sizes
        dropout: dropout prob applied after ReLU (0.0 disables)
        init: one of {"kaiming", "xavier", "none"} weight init schemes
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 343,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.0,
        init: str = "kaiming",
    ):
        super().__init__()

        layers = [
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [nn.Linear(hidden2, num_classes)]
        self.net = nn.Sequential(*layers)

        if init != "none":
            self._init_weights(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _init_weights(self, scheme: str = "kaiming"):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if scheme == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                else:
                    # default PyTorch init
                    pass
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)


# -------------------------
# Factories & utilities
# -------------------------

def create_model(in_dim: int, cfg=None) -> FlatClassifier:
    """
    Convenience factory. If a TrainConfig is provided, fields are used.
    """
    if cfg is None:
        return FlatClassifier(in_dim=in_dim)
    return FlatClassifier(
        in_dim=in_dim,
        num_classes=getattr(cfg, "num_classes", 343),
        hidden1=getattr(cfg, "hidden1", 512),
        hidden2=getattr(cfg, "hidden2", 128),
        dropout=getattr(cfg, "dropout", 0.0),
        init="kaiming",
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
