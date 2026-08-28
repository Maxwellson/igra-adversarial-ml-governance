"""Technical-risk measurement on WDBC (Section IX-A, Table IV).

Trains the Section IX-A classifier for five independent seeds (42-46),
runs FGSM, 20-step PGD (3 restarts), and random label-flipping poisoning
at strengths {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}, and reports PAVI as
mean +/- SD across seeds, with each seed's PAVI computed relative to that
same seed's own clean-model accuracy.

Writes results/raw/technical_wdbc.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from attacks.attacks import fgsm_attack, label_flip_poison, pgd_attack  # noqa: E402
from experiments.common import (accuracy, load_wdbc_split, pavi,  # noqa: E402
                                train_model)

SEEDS = [42, 43, 44, 45, 46]
PRIMARY_SEED = 42
STRENGTHS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def run():
    per_seed = {}
    for seed in SEEDS:
        x_tr, x_te, y_tr, y_te = load_wdbc_split(seed)
        model = train_model(x_tr, y_tr, seed)
        clean_acc = accuracy(model, x_te, y_te)

        fgsm_pavi, pgd_pavi, poison_pavi = [], [], []
        for eps in STRENGTHS:
            x_adv = fgsm_attack(model, x_te, y_te, eps)
            fgsm_pavi.append(pavi(clean_acc, accuracy(model, x_adv, y_te)))

            x_adv = pgd_attack(model, x_te, y_te, eps, x_tr,
                               seed=seed * 1000 + int(eps * 100))
            pgd_pavi.append(pavi(clean_acc, accuracy(model, x_adv, y_te)))

        for frac in STRENGTHS:
            y_poisoned = label_flip_poison(
                y_tr, frac, seed=seed * 100 + int(frac * 100))
            poisoned_model = train_model(x_tr, y_poisoned, seed)
            poison_pavi.append(
                pavi(clean_acc, accuracy(poisoned_model, x_te, y_te)))

        per_seed[seed] = {
            "clean_acc": clean_acc,
            "fgsm_pavi": fgsm_pavi,
            "pgd_pavi": pgd_pavi,
            "poison_pavi": poison_pavi,
        }
        print(f"seed {seed}: clean acc {clean_acc:.4f}")

    clean_accs = np.array([per_seed[s]["clean_acc"] for s in SEEDS])
    table4 = {}
    for name in ["fgsm_pavi", "pgd_pavi", "poison_pavi"]:
        arr = np.array([per_seed[s][name] for s in SEEDS])  # seeds x eps
        table4[name] = {
            "mean": arr.mean(axis=0).tolist(),
            "sd": arr.std(axis=0).tolist(),  # population SD, matching the paper
        }

    # Pool of all seed-level PAVI values across attacks and strengths,
    # used as the technical_risk sampling pool in Section XI-D.
    pool = []
    for s in SEEDS:
        for name in ["fgsm_pavi", "pgd_pavi", "poison_pavi"]:
            pool.extend(per_seed[s][name])

    results = {
        "seeds": SEEDS,
        "strengths": STRENGTHS,
        "primary_seed_clean_acc": per_seed[PRIMARY_SEED]["clean_acc"],
        "clean_acc_mean": float(clean_accs.mean()),
        "clean_acc_sd": float(clean_accs.std()),  # population SD
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "table4": table4,
        "technical_pool": pool,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "technical_wdbc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPrimary-seed clean acc: {results['primary_seed_clean_acc']:.4f}")
    print(f"Across-seed clean acc: {results['clean_acc_mean']*100:.2f}% "
          f"+/- {results['clean_acc_sd']*100:.2f}%")
    print("\nTable IV (mean +/- SD, n=5):")
    print(f"{'eps/frac':>9} {'FGSM':>16} {'PGD':>16} {'Poison':>16}")
    for i, eps in enumerate(STRENGTHS):
        row = [f"{table4[n]['mean'][i]:.3f} +/- {table4[n]['sd'][i]:.3f}"
               for n in ['fgsm_pavi', 'pgd_pavi', 'poison_pavi']]
        print(f"{eps:>9.2f} {row[0]:>16} {row[1]:>16} {row[2]:>16}")
    return results


if __name__ == "__main__":
    run()
