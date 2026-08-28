"""Fairness-signal integration into IGRA (Section XI-D, Table XI).

Defines ethical_risk = min(1, |EOD| / delta_E) with delta_E = 0.10 (an
illustrative policy threshold), using absolute EOD as a state-based
signal. For each poisoning fraction, the measured mean EOD from Table V
is mapped through this definition and combined, over 500 simulated
trials, with technical_risk sampled from the pool of all seed-level WDBC
PAVI measurements (Table IV) and simulated low-band accountability_risk,
using context_weight = 1.30. Escalation = governance_risk >= 0.50.

Also propagates EOD seed by seed at full poisoning strength, counting the
independently trained models whose ethical channel alone crosses the
human-review boundary (ethical_risk > 0.50 / 1.30 = 0.385).

Requires results/raw/fairness.json and results/raw/technical_wdbc.json.
Writes results/raw/fairness_integration.json.
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
N_TRIALS = 500
DELTA_E = 0.10
CONTEXT_WEIGHT = 1.30
SIGMA_SUB = 0.05
N_SUB = 3
ACC_LOW = (0.00, 0.15)
FRACTIONS = ["0.0", "0.25", "0.5", "0.75", "1.0"]


def run():
    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "results", "raw")
    with open(os.path.join(raw_dir, "fairness.json")) as f:
        fair = json.load(f)
    with open(os.path.join(raw_dir, "technical_wdbc.json")) as f:
        tech = json.load(f)

    pool = np.array(tech["technical_pool"])
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "n_trials": N_TRIALS, "delta_E": DELTA_E,
               "table11": {}}

    for frac in FRACTIONS:
        mean_eod = fair["table5"][frac]["eod"]
        ethical_risk = min(1.0, abs(mean_eod) / DELTA_E)

        t = rng.choice(pool, size=N_TRIALS, replace=True)
        a_true = rng.uniform(*ACC_LOW, N_TRIALS)
        a_obs = np.clip(a_true[:, None]
                        + rng.normal(0.0, SIGMA_SUB, (N_TRIALS, N_SUB)),
                        0, 1).mean(axis=1)
        e = np.full(N_TRIALS, ethical_risk)

        igra_esc = igra_escalates(t, e, a_obs, cw=CONTEXT_WEIGHT)
        pavi_esc = pavi_only_escalates(t, cw=CONTEXT_WEIGHT)

        results["table11"][frac] = {
            "mean_eod": mean_eod,
            "ethical_risk": ethical_risk,
            "igra_escalation": float(igra_esc.mean()),
            "pavi_escalation": float(pavi_esc.mean()),
        }

    # Seed-level propagation at full poisoning strength.
    eods = np.array(fair["table5"]["1.0"]["eod_per_seed"])
    boundary = 0.50 / CONTEXT_WEIGHT
    seed_risks = np.minimum(1.0, np.abs(eods) / DELTA_E)
    crossings = int((seed_risks > boundary).sum())
    results["seed_level"] = {
        "eod_per_seed": eods.tolist(),
        "eod_min": float(eods.min()),
        "eod_max": float(eods.max()),
        "boundary_ethical_risk": boundary,
        "crossings": crossings,
        "n_seeds": len(eods),
    }

    out = os.path.join(raw_dir, "fairness_integration.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("Table XI:")
    print(f"{'frac':>6} {'meanEOD':>8} {'eth_risk':>9} {'IGRA esc':>9} "
          f"{'PAVI esc':>9}")
    for frac in FRACTIONS:
        r = results["table11"][frac]
        print(f"{float(frac):>6.2f} {r['mean_eod']:>8.3f} "
              f"{r['ethical_risk']:>9.3f} {r['igra_escalation']:>9.1%} "
              f"{r['pavi_escalation']:>9.1%}")
    sl = results["seed_level"]
    print(f"\nSeed-level at full strength: EOD {sl['eod_min']:.3f} to "
          f"{sl['eod_max']:.3f}; crossings {sl['crossings']}/{sl['n_seeds']} "
          f"(ethical_risk > {sl['boundary_ethical_risk']:.3f})")
    return results


if __name__ == "__main__":
    run()
