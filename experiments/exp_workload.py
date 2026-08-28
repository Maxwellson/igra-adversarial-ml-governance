"""Review-workload experiment (Section XI-C, Table X).

Precision, recall, and reviews per 1000 monitored incidents at
critical-event prevalence in {1%, 5%, 10%, 25%, 50%}, using the same
observation model as the borderline/noisy scenario (shared sigma = 0.08
plus independent sigma = 0.03 over three sub-signals; technical severity
low, in [0.00, 0.15], observed exactly).

This experiment deliberately isolates incidents whose dominant risk is
non-technical: a critical incident places one non-technical pillar
(ethical or accountability, chosen at random) in the high band
[0.45, 0.85] with the other in [0.00, 0.25]; a non-critical incident
draws both non-technical pillars from [0.00, 0.25]. PAVI-only therefore
never escalates here by construction, which demonstrates structural
blindness rather than a general production comparison (the paper states
this scope explicitly).

n = 5000 incidents per prevalence level. Writes
results/raw/workload.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from src.igra import igra_escalates, pavi_only_escalates  # noqa: E402

SEED = 42
N_PER_LEVEL = 5000
PREVALENCES = [0.01, 0.05, 0.10, 0.25, 0.50]
SIGMA_SHARED = 0.08
SIGMA_INDEP = 0.03
N_SUB = 3
HIGH = (0.45, 0.85)
BACKGROUND = (0.00, 0.25)


def observe(rng, true_vals):
    n = len(true_vals)
    shared = rng.normal(0.0, SIGMA_SHARED, n)
    subs = np.clip(true_vals[:, None] + shared[:, None]
                   + rng.normal(0.0, SIGMA_INDEP, (n, N_SUB)), 0, 1)
    return subs.mean(axis=1)


def run():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "n_per_level": N_PER_LEVEL, "table10": {}}

    for prev in PREVALENCES:
        n = N_PER_LEVEL
        critical = rng.random(n) < prev
        t = rng.uniform(0.00, 0.15, n)
        e = rng.uniform(*BACKGROUND, n)
        a = rng.uniform(*BACKGROUND, n)
        driver_is_ethical = rng.random(n) < 0.5
        high_draw = rng.uniform(*HIGH, n)
        e = np.where(critical & driver_is_ethical, high_draw, e)
        a = np.where(critical & ~driver_is_ethical, high_draw, a)

        e_obs = observe(rng, e)
        a_obs = observe(rng, a)

        igra_esc = igra_escalates(t, e_obs, a_obs)
        pavi_esc = pavi_only_escalates(t)

        row = {}
        for name, esc in (("pavi", pavi_esc), ("igra", igra_esc)):
            tp = int((esc & critical).sum())
            fp = int((esc & ~critical).sum())
            fn = int((~esc & critical).sum())
            row[name] = {
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "reviews_per_1000": 1000.0 * (tp + fp) / n,
            }
        results["table10"][str(prev)] = row

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "workload.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("Table X:")
    print(f"{'prev':>6} {'policy':>10} {'precision':>10} {'recall':>8} "
          f"{'rev/1000':>9}")
    for prev in PREVALENCES:
        for name in ("pavi", "igra"):
            r = results["table10"][str(prev)][name]
            print(f"{prev:>6.0%} {name.upper():>10} {r['precision']:>10.3f} "
                  f"{r['recall']:>8.3f} {r['reviews_per_1000']:>9.1f}")
    return results


if __name__ == "__main__":
    run()
