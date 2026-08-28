"""Attack implementations (Section IX-A of the paper).

FGSM and a strengthened 20-step PGD under an L-inf threat model operating
directly on the standardized 30-dimensional feature vector (not pixels),
plus random label-flipping poisoning. Perturbed inputs are clipped to the
training set's observed per-feature range so values stay within the
standardized feature space actually seen during training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cpu")

PGD_STEPS = 20
PGD_RESTARTS = 3


def _feature_bounds(x_train: np.ndarray):
    lo = torch.tensor(x_train.min(axis=0), dtype=torch.float32, device=DEVICE)
    hi = torch.tensor(x_train.max(axis=0), dtype=torch.float32, device=DEVICE)
    return lo, hi


def fgsm_attack(model, x: np.ndarray, y: np.ndarray,
                eps: float) -> np.ndarray:
    """Single-step FGSM: x_adv = x + eps * sign(grad_x loss).

    Per Section IX-A, the training-range clipping applies to PGD; FGSM is
    the plain single-step attack."""
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE,
                      requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    loss = nn.CrossEntropyLoss()(model(xt), yt)
    grad = torch.autograd.grad(loss, xt)[0]
    x_adv = xt.detach() + eps * grad.sign()
    return x_adv.cpu().numpy()


def pgd_attack(model, x: np.ndarray, y: np.ndarray, eps: float,
               x_train: np.ndarray, seed: int,
               steps: int = PGD_STEPS,
               restarts: int = PGD_RESTARTS) -> np.ndarray:
    """20-step PGD, step size eps/4, random initialization within the
    eps-ball, 3 restarts per sample keeping the highest-loss adversarial
    example, projection onto the eps-ball after every step, and clipping
    to the training set's observed per-feature range."""
    lo, hi = _feature_bounds(x_train)
    alpha = eps / 4.0
    torch.manual_seed(seed)
    x0 = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    best_adv = x0.clone()
    best_loss = torch.full((x0.shape[0],), -np.inf, device=DEVICE)

    for _ in range(restarts):
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = torch.min(torch.max(x0 + delta, lo), hi)
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            losses = loss_fn(model(x_adv), yt)
            grad = torch.autograd.grad(losses.sum(), x_adv)[0]
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps)  # project
            x_adv = torch.min(torch.max(x_adv, lo), hi)              # clip
        with torch.no_grad():
            final_losses = loss_fn(model(x_adv), yt)
        improved = final_losses > best_loss
        best_adv[improved] = x_adv[improved].detach()
        best_loss[improved] = final_losses[improved]

    return best_adv.cpu().numpy()


def label_flip_poison(y_train: np.ndarray, fraction: float,
                      seed: int) -> np.ndarray:
    """Random label-flipping: flip round(fraction * n) labels chosen
    uniformly without replacement (Section IX-A)."""
    rng = np.random.default_rng(seed)
    y_poisoned = y_train.copy()
    n_flip = int(round(fraction * len(y_train)))
    if n_flip > 0:
        idx = rng.choice(len(y_train), size=n_flip, replace=False)
        y_poisoned[idx] = 1 - y_poisoned[idx]
    return y_poisoned
