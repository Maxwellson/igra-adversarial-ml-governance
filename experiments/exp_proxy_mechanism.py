"""Follow-up to exp_proxy_fairness.py: mechanism checks and a model-based attacker.

Same data, seeds (42..81), proxy strengths and models as exp_proxy_fairness.py.
For every trained model this script records, in addition to accuracy and group
rates:
  * a permutation null for the signed equal-opportunity gap (group labels
    shuffled among the positive test cases, 2,000 shuffles), giving a
    model-specific 95th-percentile threshold and a permutation p-value;
  * the equal-opportunity gap after the proxy input is shuffled across the test
    set (mean of 5 shuffles) and after it is set to its training mean, to test
    whether the group effect runs through the proxy;
and it adds a second attacker that ranks the group-1 positive training examples
by distance to the CLEAN MODEL's decision boundary (logit margin) instead of the
generator's latent boundary.

Writes results/raw/proxy_mechanism.json.
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
from experiments.common import MLP, accuracy, predict, train_model  # noqa: E402
from experiments.exp_proxy_fairness import (FRACTIONS, PROXY_D, SEEDS,  # noqa: E402
                                            add_proxy, group_stats)

N_PERM = 2000
N_SHUF = 5


def eog_from_pred(yte, pred, gte):
    _, tpr = group_stats(yte, pred, gte)
    return tpr[0] - tpr[1]


def perm_null(yte, pred, gte, rng):
    pos = yte == 1
    a = pred[pos].astype(float)
    gp = gte[pos]
    obs = a[gp == 0].mean() - a[gp == 1].mean()
    G = rng.permuted(np.tile(gp, (N_PERM, 1)), axis=1)
    n1 = G.sum(1); s1 = (G * a).sum(1)
    ok = (n1 > 0) & (n1 < len(gp))
    diff = (a.sum() - s1[ok]) / (len(gp) - n1[ok]) - s1[ok] / n1[ok]
    p = (1 + np.sum(np.abs(diff) >= abs(obs))) / (len(diff) + 1)
    return float(obs), float(np.quantile(np.abs(diff), 0.95)), float(p)


def record(model, xte, yte, gte, seed, d, rng):
    pred = predict(model, xte)
    sel, tpr = group_stats(yte, pred, gte)
    obs, null95, p = perm_null(yte, pred, gte, rng)
    out = {"acc": accuracy(model, xte, yte), "sel0": sel[0], "sel1": sel[1],
           "tpr0": tpr[0], "tpr1": tpr[1], "null95": null95, "perm_p": p}
    if d > 0:
        e = []
        for _ in range(N_SHUF):
            xs = xte.copy(); xs[:, -1] = rng.permutation(xs[:, -1])
            e.append(eog_from_pred(yte, predict(model, xs), gte))
        xz = xte.copy(); xz[:, -1] = 0.0
        out["abl_shuffle_eod"] = float(np.mean(e))
        out["abl_zero_eod"] = eog_from_pred(yte, predict(model, xz), gte)
    return out


@torch.no_grad()
def margins(model, x):
    z = model(torch.tensor(x, dtype=torch.float32))
    return (z[:, 1] - z[:, 0]).numpy()


def run():
    out = {"proxy_d": PROXY_D, "seeds": SEEDS, "fractions": FRACTIONS,
           "n_perm": N_PERM, "n_shuffle": N_SHUF, "runs": {}}
    t0 = time.time()
    for d in PROXY_D:
        out["runs"][str(d)] = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 100 + int(d * 10))
            x, y, g, logit, med = make_dataset(seed)
            x = add_proxy(x, g, d, seed)
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
            sc = StandardScaler().fit(x[tr])
            xtr, xte = sc.transform(x[tr]), sc.transform(x[te])
            ytr, yte, gtr, gte = y[tr], y[te], g[tr], g[te]
            target = np.where((gtr == 1) & (ytr == 1))[0]
            order_lat = target[np.argsort(np.abs(logit[tr][target] - med))]
            clean = train_model(xtr, ytr, seed)
            order_mod = target[np.argsort(np.abs(margins(clean, xtr)[target]))]
            per = {"latent": {}, "model": {}, "n_target": int(len(target)), "n_train": int(len(tr))}
            c = record(clean, xte, yte, gte, seed, d, rng)
            per["latent"]["0.0"] = c; per["model"]["0.0"] = c
            for name, order in (("latent", order_lat), ("model", order_mod)):
                for frac in FRACTIONS[1:]:
                    yp = ytr.copy(); yp[order[:int(round(frac * len(order)))]] = 0
                    per[name][str(frac)] = record(train_model(xtr, yp, seed), xte, yte, gte, seed, d, rng)
            out["runs"][str(d)][str(seed)] = per
        print(f"d={d}: done ({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw", "proxy_mechanism.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    run()
