"""Proxy-strength study: can label-flip poisoning create group disparity?

Extends the Section IX-B fairness experiment with one extra model input, a
proxy feature  proxy = d * g + Normal(0, 1)  that is correlated with the
protected attribute g but carries no information about the label. d is the
standardized separation between the groups on the proxy (d = 0 is the control
that reproduces the original design, where the model has no access to any
group information).

For each proxy strength d, each of N_SEEDS independent seeds, and each
poisoning fraction, the boundary-targeted attack of exp_fairness.py is applied
(flip the labels of the closest fraction of the g = 1, y = 1 training examples
from 1 to 0) and a model is retrained. Recorded per model: accuracy, group
true-positive rates and selection rates, so that SIGNED gaps
(group 0 minus group 1) can be analyzed in addition to absolute ones.

Writes results/raw/proxy_fairness.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_generation.generate_synthetic_fairness import make_dataset  # noqa: E402
from experiments.common import accuracy, predict, train_model  # noqa: E402

PROXY_D = [0.0, 0.5, 1.0, 2.0]
N_SEEDS = int(os.environ.get("PROXY_N_SEEDS", 40))
SEEDS = list(range(42, 42 + N_SEEDS))
FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]


def add_proxy(x, g, d, seed):
    rng = np.random.default_rng(seed + 10_000)  # separate stream: base data unchanged
    proxy = d * g + rng.normal(0.0, 1.0, len(g))
    return np.column_stack([x, proxy])


def group_stats(y_true, y_pred, g):
    sel = [float(np.mean(y_pred[g == v])) for v in (0, 1)]
    tpr = [float(np.mean(y_pred[(g == v) & (y_true == 1)])) for v in (0, 1)]
    return sel, tpr


def run():
    out = {"proxy_d": PROXY_D, "seeds": SEEDS, "fractions": FRACTIONS, "runs": {}}
    t0 = time.time()
    for d in PROXY_D:
        out["runs"][str(d)] = {}
        for seed in SEEDS:
            x, y, g, logit, med = make_dataset(seed)
            x = add_proxy(x, g, d, seed)
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
            scaler = StandardScaler().fit(x[tr])
            xtr, xte = scaler.transform(x[tr]), scaler.transform(x[te])
            ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]
            target = np.where((gtr == 1) & (ytr == 1))[0]
            order = target[np.argsort(np.abs(logit[tr][target] - med))]
            per = {}
            for frac in FRACTIONS:
                n_flip = int(round(frac * len(order)))
                yp = ytr.copy(); yp[order[:n_flip]] = 0
                model = train_model(xtr, yp, seed)
                pred = predict(model, xte)
                sel, tpr = group_stats(yte, pred, gte)
                per[str(frac)] = {"acc": accuracy(model, xte, yte),
                                  "sel0": sel[0], "sel1": sel[1],
                                  "tpr0": tpr[0], "tpr1": tpr[1]}
            out["runs"][str(d)][str(seed)] = per
        print(f"d={d}: done ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "results", "raw", "proxy_fairness.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    run()
