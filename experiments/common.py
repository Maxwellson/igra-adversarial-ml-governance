"""Shared model, training, and data utilities (Section IX-A of the paper).

Architecture and training procedure used for BOTH datasets: a feedforward
classifier with hidden layers of 32 and 16 units (ReLU) and a 2-unit output
layer, Adam (lr = 0.005, weight decay = 1e-4), 150 epochs, cross-entropy
loss, full-batch updates. The input layer is sized to each dataset's own
dimensionality (30 for WDBC, 8 for the synthetic fairness dataset).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LR = 0.005
WEIGHT_DECAY = 1e-4
EPOCHS = 150
TEST_SIZE = 0.30

torch.set_num_threads(1)  # single-threaded for cross-machine determinism
DEVICE = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_model(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> MLP:
    """Train the Section IX-A classifier deterministically for a given seed."""
    torch.manual_seed(seed)
    model = MLP(x_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    preds = model(xt).argmax(dim=1).cpu().numpy()
    return float((preds == y).mean())


@torch.no_grad()
def predict(model: MLP, x: np.ndarray) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    return model(xt).argmax(dim=1).cpu().numpy()


def pavi(clean_acc: float, attacked_acc: float) -> float:
    """Post-Attack Vulnerability Index: relative accuracy degradation,
    clipped below at 0 (Section VIII-A)."""
    return max(0.0, (clean_acc - attacked_acc) / clean_acc)


def load_wdbc_split(seed: int):
    """WDBC (569 x 30), 70/30 stratified split, standardized on the
    training split only (Section IX-A)."""
    x, y = load_breast_cancer(return_X_y=True)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(x_tr)
    return (scaler.transform(x_tr), scaler.transform(x_te),
            y_tr.astype(np.int64), y_te.astype(np.int64))
