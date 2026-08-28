"""Borderline/noisy scenario (Section X-E, Table VIII), threshold
sensitivity sweep (Section X-D, Fig. 4), alternative policy oracles
(Section X-F, Table IX), the pure-max comparison, and the mean-branch
disambiguation scenario (Section X-C).

Borderline generator (n = 1000): true ethical and accountability severity
drawn continuously from [0.30, 0.60] (straddling the 0.45 critical
cutoff); technical severity is held low by construction (drawn from
[0.00, 0.15]) and, being the measured PAVI, is observed exactly.
Ethical and accountability channels are observed through three noisy
sub-signals with correlated observation noise: a per-incident shared
component (sigma = 0.08) plus independent per-sub-signal noise
(sigma = 0.03), clipped to [0, 1].

The same frozen draw of this scenario's underlying random severity values
is reused for the threshold sweep, the policy oracles, and the pure-max
comparison, exactly as Section X states.

Writes results/raw/borderline.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from src.igra import (igra_escalates, igra_governance,  # noqa: E402
                      pavi_only_escalates, pure_max_escalates,
                      pure_max_governance, response_tier, wilson_ci)

SEED = 42
N_TRIALS = 1000
CRITICAL_CUTOFF = 0.45
SIGMA_SHARED = 0.08
SIGMA_INDEP = 0.03
N_SUB = 3

MEANBRANCH_SEED = 43
MEANBRANCH_N = 2000


def generate_borderline(rng, n=N_TRIALS):
    t = rng.uniform(0.00, 0.15, n)
    e = rng.uniform(0.30, 0.60, n)
    a = rng.uniform(0.30, 0.60, n)
    critical = (e > CRITICAL_CUTOFF) | (a > CRITICAL_CUTOFF)

    shared = rng.normal(0.0, SIGMA_SHARED, n)
    e_obs = np.clip(e[:, None] + shared[:, None]
                    + rng.normal(0.0, SIGMA_INDEP, (n, N_SUB)),
                    0, 1).mean(axis=1)
    a_obs = np.clip(a[:, None] + shared[:, None]
                    + rng.normal(0.0, SIGMA_INDEP, (n, N_SUB)),
                    0, 1).mean(axis=1)
    return dict(t=t, e=e, a=a, critical=critical,
                t_obs=t, e_obs=e_obs, a_obs=a_obs)


def confusion(esc, critical):
    tp = int((esc & critical).sum())
    fp = int((esc & ~critical).sum())
    fn = int((~esc & critical).sum())
    tn = int((~esc & ~critical).sum())
    n = len(critical)
    n_crit, n_non = tp + fn, fp + tn
    return {
        "accuracy": (tp + tn) / n,
        "accuracy_ci": wilson_ci(tp + tn, n),
        "false_escalation": fp / n_non if n_non else 0.0,
        "false_escalation_ci": wilson_ci(fp, n_non),
        "missed_critical": fn / n_crit if n_crit else 0.0,
        "missed_critical_ci": wilson_ci(fn, n_crit),
        "balanced_accuracy": 0.5 * ((tp / n_crit if n_crit else 0.0)
                                    + (tn / n_non if n_non else 0.0)),
        "escalation_rate": (tp + fp) / n,
    }


def run():
    rng = np.random.default_rng(SEED)
    s = generate_borderline(rng)

    results = {"seed": SEED, "n_trials": N_TRIALS,
               "critical_prevalence": float(s["critical"].mean())}

    # --- Table VIII -------------------------------------------------------
    pavi_esc = pavi_only_escalates(s["t_obs"])
    igra_esc = igra_escalates(s["t_obs"], s["e_obs"], s["a_obs"])
    results["table8"] = {
        "pavi": confusion(pavi_esc, s["critical"]),
        "igra": confusion(igra_esc, s["critical"]),
    }

    # --- Pure max on the frozen borderline draw (Section X-C) -------------
    pmax_esc = pure_max_escalates(s["t_obs"], s["e_obs"], s["a_obs"])
    results["pure_max_borderline"] = {
        "identical_to_igra": bool(np.array_equal(igra_esc, pmax_esc)),
        "accuracy": confusion(pmax_esc, s["critical"])["accuracy"],
    }

    # --- Threshold sweep (Section X-D, Fig. 4) ----------------------------
    taus = [round(0.05 * k, 2) for k in range(1, 13)]  # 0.05 .. 0.60
    sweep = []
    for tau in taus:
        esc = igra_escalates(s["t_obs"], s["e_obs"], s["a_obs"], tau=tau)
        c = confusion(esc, s["critical"])
        sweep.append({"tau": tau, "accuracy": c["accuracy"],
                      "balanced_accuracy": c["balanced_accuracy"],
                      "false_escalation": c["false_escalation"],
                      "missed_critical": c["missed_critical"]})
    results["threshold_sweep"] = sweep

    # --- Alternative policy oracles (Section X-F, Table IX) ---------------
    oracles = {
        "max_original": s["critical"],
        "weighted_sum": (0.4 * s["t"] + 0.3 * s["e"] + 0.3 * s["a"]) > 0.45,
        "lexicographic": (s["t"] > 0.50) | (s["e"] > 0.50) | (s["a"] > 0.50),
        "cost_sensitive": (s["t"] > 0.40) | (s["e"] > 0.55) | (s["a"] > 0.40),
    }
    results["table9"] = {}
    for name, crit in oracles.items():
        results["table9"][name] = {
            "pavi_acc": float((pavi_esc == crit).mean()),
            "igra_acc": float((igra_esc == crit).mean()),
            "critical_rate": float(np.mean(crit)),
        }

    # --- Mean-branch disambiguation scenario (Section X-C) ----------------
    rng2 = np.random.default_rng(MEANBRANCH_SEED)
    n2 = MEANBRANCH_N
    e2 = rng2.uniform(0.12, 0.19, n2)       # near but below tau
    t2 = rng2.uniform(0.00, 0.08, n2)
    a2 = rng2.uniform(0.00, 0.08, n2)
    crit2 = np.zeros(n2, dtype=bool)        # nothing exceeds 0.45
    g_igra = igra_governance(t2, e2, a2)
    g_pmax = pure_max_governance(t2, e2, a2)
    tier_igra = response_tier(g_igra)
    tier_pmax = response_tier(g_pmax)
    results["mean_branch"] = {
        "n": n2,
        "tier_disagreement_rate": float((tier_igra != tier_pmax).mean()),
        "igra_always_continue": bool((tier_igra == 0).all()),
        "pmax_tiers": {str(k): int((tier_pmax == k).sum())
                       for k in np.unique(tier_pmax)},
        "igra_acc": float(((g_igra >= 0.5) == crit2).mean()),
        "pmax_acc": float(((g_pmax >= 0.5) == crit2).mean()),
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "results", "raw", "borderline.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"critical prevalence: {results['critical_prevalence']:.3f}")
    for pol in ("pavi", "igra"):
        c = results["table8"][pol]
        print(f"{pol.upper():>5}: acc {c['accuracy']:.3f} "
              f"[{c['accuracy_ci'][0]:.3f},{c['accuracy_ci'][1]:.3f}]  "
              f"false-esc {c['false_escalation']:.3f} "
              f"[{c['false_escalation_ci'][0]:.3f},"
              f"{c['false_escalation_ci'][1]:.3f}]  "
              f"missed {c['missed_critical']:.3f} "
              f"[{c['missed_critical_ci'][0]:.3f},"
              f"{c['missed_critical_ci'][1]:.3f}]  "
              f"bal-acc {c['balanced_accuracy']:.3f}  "
              f"esc-rate {c['escalation_rate']:.3f}")
    print("\nThreshold sweep:")
    for r in sweep:
        print(f"  tau {r['tau']:.2f}: acc {r['accuracy']:.3f} "
              f"bal {r['balanced_accuracy']:.3f} "
              f"fe {r['false_escalation']:.3f} "
              f"mc {r['missed_critical']:.3f}")
    print("\nTable IX:")
    for name, r in results["table9"].items():
        print(f"  {name:<15} PAVI {r['pavi_acc']:.3f}  "
              f"IGRA {r['igra_acc']:.3f}  "
              f"(critical rate {r['critical_rate']:.3f})")
    mb = results["mean_branch"]
    print(f"\nMean-branch: tier disagreement {mb['tier_disagreement_rate']:.3f}, "
          f"IGRA always continue: {mb['igra_always_continue']}, "
          f"acc {mb['igra_acc']:.3f} vs {mb['pmax_acc']:.3f}")
    print("Pure-max identical on borderline:",
          results["pure_max_borderline"]["identical_to_igra"],
          f"(acc {results['pure_max_borderline']['accuracy']:.3f})")
    return results


if __name__ == "__main__":
    run()
