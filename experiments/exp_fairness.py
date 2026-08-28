"""Fairness-targeted poisoning experiment (Sections IV-A and IX-B, Table V).

For each of 8 independent seeds (42-49): generate the controlled synthetic
fairness dataset, split 70/30 stratified by label, standardize on the
training split only, train the Section IX-A classifier, then apply the
boundary-targeted fairness poisoning attack at fractions
{0.00, 0.25, 0.50, 0.75, 1.00}: select training examples with g = 1 and
y = 1, rank them by |logit - median(logit)| ascending (the synthetic
generator's latent score, not the trained classifier's decision function),
flip the label of the closest X% of that subset from 1 to 0, and retrain
with the same architecture, optimizer, and epoch count.

Reported per fraction, averaged across seeds: clean-test accuracy, PAVI
relative to the same seed's clean (fraction 0) model, demographic parity
difference (DPD), and equal-opportunity difference (EOD) with a 95% t CI
across seeds.

Writes results/raw/fairness.json (including per-seed EOD values, which
Section XI-D consumes).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from data_generation.generate_synthetic_fairness import make_dataset  # noqa: E402
from experiments.common import accuracy, pavi, predict, train_model  # noqa: E402

SEEDS = list(range(42, 50))  # 42..49
FRACTIONS = [0.00, 0.25, 0.50, 0.75, 1.00]


def group_metrics(y_true, y_pred, g):
    """Demographic parity difference and equal-opportunity difference."""
    sel = [np.mean(y_pred[g == v]) for v in (0, 1)]
    dpd = abs(sel[0] - sel[1])
    tpr = [np.mean(y_pred[(g == v) & (y_true == 1)]) for v in (0, 1)]
    eod = abs(tpr[0] - tpr[1])
    return float(dpd), float(eod)


def run():
    per_seed = {s: {} for s in SEEDS}
    for seed in SEEDS:
        x, y, g, logit, median_logit = make_dataset(seed)
        idx = np.arange(len(y))
        idx_tr, idx_te = train_test_split(
            idx, test_size=0.30, stratify=y, random_state=seed)
        x_tr, x_te = x[idx_tr], x[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        g_tr, g_te = g[idx_tr], g[idx_te]
        logit_tr = logit[idx_tr]

        scaler = StandardScaler().fit(x_tr)
        x_tr_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_te)

        # Targeted subset: g = 1 and y = 1, ranked by distance from the
        # label-generating boundary, ascending (closest first).
        target = np.where((g_tr == 1) & (y_tr == 1))[0]
        order = target[np.argsort(np.abs(logit_tr[target] - median_logit))]

        clean_acc = None
        for frac in FRACTIONS:
            n_flip = int(round(frac * len(order)))
            y_poisoned = y_tr.copy()
            y_poisoned[order[:n_flip]] = 0
            model = train_model(x_tr_s, y_poisoned, seed)
            acc = accuracy(model, x_te_s, y_te)
            if frac == 0.0:
                clean_acc = acc
            y_pred = predict(model, x_te_s)
            dpd, eod = group_metrics(y_te, y_pred, g_te)
            per_seed[seed][frac] = {
                "acc": acc,
                "pavi": pavi(clean_acc, acc),
                "dpd": dpd,
                "eod": eod,
            }
        print(f"seed {seed}: clean acc {clean_acc:.3f}, "
              f"EOD@1.0 {per_seed[seed][1.0]['eod']:.3f}")

    table5 = {}
    for frac in FRACTIONS:
        accs = np.array([per_seed[s][frac]["acc"] for s in SEEDS])
        pavis = np.array([per_seed[s][frac]["pavi"] for s in SEEDS])
        dpds = np.array([per_seed[s][frac]["dpd"] for s in SEEDS])
        eods = np.array([per_seed[s][frac]["eod"] for s in SEEDS])
        n = len(eods)
        sem = eods.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        table5[frac] = {
            "acc": float(accs.mean()),
            "pavi": float(pavis.mean()),
            "dpd": float(dpds.mean()),
            "eod": float(eods.mean()),
            "eod_ci": [max(0.0, float(eods.mean() - tcrit * sem)),
                       float(eods.mean() + tcrit * sem)],
            "eod_per_seed": eods.tolist(),
        }

    results = {"seeds": SEEDS, "fractions": FRACTIONS,
               "per_seed": {str(k): {str(f): v for f, v in d.items()}
                            for k, d in per_seed.items()},
               "table5": {str(k): v for k, v in table5.items()}}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "fairness.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nTable V (n=8 seeds):")
    print(f"{'frac':>6} {'Acc.':>7} {'PAVI':>7} {'DPD':>7} "
          f"{'EOD [95% CI]':>22}")
    for frac in FRACTIONS:
        t = table5[frac]
        print(f"{frac:>6.2f} {t['acc']:>7.3f} {t['pavi']:>7.3f} "
              f"{t['dpd']:>7.3f}   {t['eod']:.3f} "
              f"[{t['eod_ci'][0]:.3f},{t['eod_ci'][1]:.3f}]")
    eods_full = np.array(table5[1.0]["eod_per_seed"])
    print(f"\nEOD range at full strength: {eods_full.min():.3f} to "
          f"{eods_full.max():.3f}")
    boundary = 0.50 / 1.30
    crossings = int(np.sum(eods_full / 0.10 > boundary))
    print(f"Seed-level crossings (ethical_risk > {boundary:.3f}): "
          f"{crossings}/8")
    return results


if __name__ == "__main__":
    run()
