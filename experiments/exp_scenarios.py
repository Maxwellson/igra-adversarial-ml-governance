"""Clean-scenario evaluation (Section X-A, Table VI), the fusion-rule
ablation (Section X-B, Table VII), and the pure-max comparison on the
clean scenarios (Section X-C).

Six scenario types, 500 trials each, using clean, separated severity
bands: low (0.00-0.15), medium (0.15-0.35), high (0.45-0.85). True
severity is drawn per pillar; an incident is critical if any pillar's
true severity exceeds the 0.45 cutoff (Section IX-C).

Observation model: technical_risk is the measured PAVI and is observed
exactly; ethical_risk and accountability_risk are each observed through
three noisy sub-signal components (sigma = 0.05, clipped to [0, 1]) whose
mean forms the channel value (Section IX-B).

Writes results/raw/scenarios.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from src.igra import (igra_escalates, naive_avg_escalates,  # noqa: E402
                      pavi_only_escalates, pure_max_escalates, wilson_ci)

SEED = 42
N_TRIALS = 500
SIGMA_SUB = 0.05
N_SUB = 3
CRITICAL_CUTOFF = 0.45

LOW = (0.00, 0.15)
MED = (0.15, 0.35)
HIGH = (0.45, 0.85)

SCENARIOS = {
    "technical_driven": (HIGH, LOW, LOW),
    "ethical_driven": (LOW, HIGH, LOW),
    "accountability_driven": (LOW, LOW, HIGH),
    "combined_moderate": (MED, MED, MED),
    "all_low": (LOW, LOW, LOW),
    "all_high": (HIGH, HIGH, HIGH),
}


def observe_channel(rng, true_vals, sigma=SIGMA_SUB, n_sub=N_SUB):
    """Channel value = mean of three noisy sub-signals, each clipped."""
    subs = np.clip(true_vals[:, None]
                   + rng.normal(0.0, sigma, size=(len(true_vals), n_sub)),
                   0.0, 1.0)
    return subs.mean(axis=1)


def generate_scenario(rng, bands, n=N_TRIALS):
    (t_band, e_band, a_band) = bands
    t = rng.uniform(*t_band, n)
    e = rng.uniform(*e_band, n)
    a = rng.uniform(*a_band, n)
    critical = (t > CRITICAL_CUTOFF) | (e > CRITICAL_CUTOFF) | \
               (a > CRITICAL_CUTOFF)
    t_obs = t  # PAVI: measured, observed exactly
    e_obs = observe_channel(rng, e)
    a_obs = observe_channel(rng, a)
    return dict(t=t, e=e, a=a, critical=critical,
                t_obs=t_obs, e_obs=e_obs, a_obs=a_obs)


def evaluate(esc, critical):
    correct = (esc == critical)
    return float(correct.mean())


def run():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "n_trials": N_TRIALS, "table6": {},
               "table7": {}, "pure_max_clean": {}}
    total_correct = {"pavi": 0, "igra": 0}

    for name, bands in SCENARIOS.items():
        s = generate_scenario(rng, bands)
        pavi_esc = pavi_only_escalates(s["t_obs"])
        igra_esc = igra_escalates(s["t_obs"], s["e_obs"], s["a_obs"])
        pmax_esc = pure_max_escalates(s["t_obs"], s["e_obs"], s["a_obs"])

        pavi_acc = evaluate(pavi_esc, s["critical"])
        igra_acc = evaluate(igra_esc, s["critical"])
        k_p = int((pavi_esc == s["critical"]).sum())
        k_i = int((igra_esc == s["critical"]).sum())
        total_correct["pavi"] += k_p
        total_correct["igra"] += k_i

        results["table6"][name] = {
            "pavi_acc": pavi_acc, "pavi_ci": wilson_ci(k_p, N_TRIALS),
            "igra_acc": igra_acc, "igra_ci": wilson_ci(k_i, N_TRIALS),
        }
        results["pure_max_clean"][name] = {
            "identical_to_igra": bool(np.array_equal(igra_esc, pmax_esc)),
            "pure_max_acc": evaluate(pmax_esc, s["critical"]),
        }
        if name in ("ethical_driven", "accountability_driven"):
            naive_esc = naive_avg_escalates(s["t_obs"], s["e_obs"],
                                            s["a_obs"])
            results["table7"][name] = {
                "dominance_acc": igra_acc,
                "naive_avg_acc": evaluate(naive_esc, s["critical"]),
            }

    n_total = N_TRIALS * len(SCENARIOS)
    for key in ("pavi", "igra"):
        k = total_correct[key]
        results["table6"].setdefault("overall", {})
        results["table6"]["overall"][f"{key}_acc"] = k / n_total
        results["table6"]["overall"][f"{key}_ci"] = wilson_ci(k, n_total)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "scenarios.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("Table VI (accuracy [95% Wilson CI]):")
    for name in list(SCENARIOS) + ["overall"]:
        r = results["table6"][name]
        print(f"  {name:<22} PAVI {r['pavi_acc']:.3f} "
              f"[{r['pavi_ci'][0]:.3f},{r['pavi_ci'][1]:.3f}]   "
              f"IGRA {r['igra_acc']:.3f} "
              f"[{r['igra_ci'][0]:.3f},{r['igra_ci'][1]:.3f}]")
    print("\nTable VII (fusion ablation):")
    for name, r in results["table7"].items():
        print(f"  {name:<22} dominance {r['dominance_acc']:.3f}   "
              f"naive-avg {r['naive_avg_acc']:.3f}")
    print("\nPure max identical to IGRA on all clean scenarios:",
          all(v["identical_to_igra"]
              for v in results["pure_max_clean"].values()))
    return results


if __name__ == "__main__":
    run()
