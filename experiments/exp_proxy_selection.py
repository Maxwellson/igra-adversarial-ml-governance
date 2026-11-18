"""Response to review: other selection rules and other decision thresholds.

Same data, seeds (42..81), proxy strengths and model as exp_proxy_fairness.py.
The main attacker flips group-1 positives in order of closeness to the label
boundary (the median of the latent score). This script adds three other
selection rules for the same flip fractions:
  * random : group-1 positives chosen uniformly at random
  * far    : group-1 positives farthest from the boundary first
  * upper  : group-1 positives nearest to the 75th percentile of the latent score
and records every model's accuracy and group true-positive rates at three
decision thresholds on P(class 1): 0.3, 0.5 (the argmax rule used in the paper),
and 0.7.

Usage: python experiments/exp_proxy_selection.py [d ...]   (default: all d)
Writes results/raw/proxy_selection_d<d>.json for each d.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_generation.generate_synthetic_fairness import make_dataset  # noqa: E402
from experiments.common import train_model  # noqa: E402
from experiments.exp_proxy_fairness import (FRACTIONS, PROXY_D, SEEDS,  # noqa: E402
                                            add_proxy, group_stats)

THRESHOLDS = [0.3, 0.5, 0.7]
RULES = ["latent", "random", "far", "upper"]


@torch.no_grad()
def prob1(model, x):
    z = model(torch.tensor(x, dtype=torch.float32))
    return torch.softmax(z, dim=1)[:, 1].numpy()


def record(model, xte, yte, gte):
    p = prob1(model, xte)
    out = {}
    for t in THRESHOLDS:
        pred = (p > t).astype(int)
        sel, tpr = group_stats(yte, pred, gte)
        out[str(t)] = {"acc": float((pred == yte).mean()), "tpr0": float(tpr[0]), "tpr1": float(tpr[1]),
                       "sel0": float(sel[0]), "sel1": float(sel[1])}
    return out


def run(ds):
    for d in ds:
        t0 = time.time()
        res = {"proxy_d": d, "seeds": SEEDS, "fractions": FRACTIONS, "thresholds": THRESHOLDS,
               "rules": RULES, "runs": {}}
        for seed in SEEDS:
            x, y, g, logit, med = make_dataset(seed)
            x = add_proxy(x, g, d, seed)
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
            sc = StandardScaler().fit(x[tr])
            xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
            ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]
            target = np.where((gtr == 1) & (ytr == 1))[0]
            q75 = float(np.quantile(logit, 0.75))
            rng = np.random.default_rng(seed * 1000 + 7)
            orders = {
                "latent": target[np.argsort(np.abs(logit[tr][target] - med))],
                "random": rng.permutation(target),
                "far": target[np.argsort(-np.abs(logit[tr][target] - med))],
                "upper": target[np.argsort(np.abs(logit[tr][target] - q75))],
            }
            per = {"clean": record(train_model(xtr, ytr, seed), xte, yte, gte)}
            for name, order in orders.items():
                per[name] = {}
                for frac in FRACTIONS[1:]:
                    yp = ytr.copy(); yp[order[:int(round(frac * len(order)))]] = 0
                    per[name][str(frac)] = record(train_model(xtr, yp, seed), xte, yte, gte)
            res["runs"][str(seed)] = per
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw",
                            f"proxy_selection_d{d}.json")
        with open(path, "w") as f:
            json.dump(res, f)
        print(f"d={d} done in {time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    ds = [float(a) for a in sys.argv[1:]] or PROXY_D
    run(ds)
