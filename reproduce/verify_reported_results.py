"""Checks results/raw/*.json against the values reported in the paper.

Every check below corresponds to a specific number in the manuscript.
Values are compared after rounding to the precision at which the paper
reports them, so this passes if and only if re-running the pipeline
reproduces the published tables digit for digit.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))
from src.igra import igra_decide  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results", "raw")

PASS, FAIL = 0, 0


def check(name, got, expected, nd=3):
    global PASS, FAIL
    if isinstance(expected, (list, tuple)):
        ok = all(round(g, nd) == round(e, nd) for g, e in zip(got, expected)) \
            and len(got) == len(expected)
    elif isinstance(expected, float):
        ok = round(got, nd) == round(expected, nd)
    else:
        ok = got == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}: got {got!r}, expected {expected!r}")


def main():
    with open(os.path.join(RAW, "technical_wdbc.json")) as f:
        t = json.load(f)
    with open(os.path.join(RAW, "fairness.json")) as f:
        fa = json.load(f)
    with open(os.path.join(RAW, "scenarios.json")) as f:
        sc = json.load(f)
    with open(os.path.join(RAW, "borderline.json")) as f:
        b = json.load(f)
    with open(os.path.join(RAW, "workload.json")) as f:
        w = json.load(f)
    with open(os.path.join(RAW, "fairness_integration.json")) as f:
        fi = json.load(f)

    # --- Section IX-A / Table IV -----------------------------------------
    check("primary-seed clean accuracy", t["primary_seed_clean_acc"],
          0.9708, nd=4)
    check("clean acc mean (%)", t["clean_acc_mean"] * 100, 96.37, nd=2)
    check("clean acc sd (%)", t["clean_acc_sd"] * 100, 0.44, nd=2)
    check("Table IV FGSM means", t["table4"]["fgsm_pavi"]["mean"],
          [0.021, 0.074, 0.126, 0.176, 0.333, 0.646])
    check("Table IV PGD means", t["table4"]["pgd_pavi"]["mean"],
          [0.021, 0.074, 0.141, 0.195, 0.375, 0.685])
    check("Table IV PGD sds", t["table4"]["pgd_pavi"]["sd"],
          [0.008, 0.020, 0.016, 0.025, 0.055, 0.038])
    check("Table IV poison means", t["table4"]["poison_pavi"]["mean"],
          [0.021, 0.062, 0.115, 0.153, 0.273, 0.469])
    check("Table IV poison sds", t["table4"]["poison_pavi"]["sd"],
          [0.018, 0.022, 0.021, 0.051, 0.040, 0.051])

    # --- Worked trace (Section VIII-F) -----------------------------------
    eth = (0.40 + 0.44 + 0.42) / 3
    acc = (0.05 + 0.04 + 0.06) / 3
    action, rec = igra_decide(0.38, eth, acc, context_weight=1.50)
    check("worked trace governance_risk", rec["governance_risk"], 0.63, nd=2)
    check("worked trace action", action, "human_review")

    # --- Table V ----------------------------------------------------------
    tv = fa["table5"]
    check("Table V acc", [tv[k]["acc"] for k in
                          ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [0.778, 0.775, 0.760, 0.740, 0.713])
    check("Table V PAVI", [tv[k]["pavi"] for k in
                           ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [0.000, 0.006, 0.024, 0.049, 0.084])
    check("Table V DPD", [tv[k]["dpd"] for k in
                          ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [0.022, 0.019, 0.023, 0.015, 0.014])
    check("Table V EOD", [tv[k]["eod"] for k in
                          ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [0.022, 0.014, 0.026, 0.027, 0.040])

    # --- Section XI-D seed-level fairness results ------------------------
    eods = tv["1.0"]["eod_per_seed"]
    check("EOD range min", min(eods), 0.008)
    check("EOD range max", max(eods), 0.096)
    check("seed-level crossings (4/8)",
          sum(1 for e in eods if e / 0.10 > 0.50 / 1.30), 4)

    # --- Table VI ---------------------------------------------------------
    t6 = sc["table6"]
    check("Table VI PAVI accs",
          [t6[k]["pavi_acc"] for k in
           ["technical_driven", "ethical_driven", "accountability_driven",
            "combined_moderate", "all_low", "all_high"]],
          [1.000, 0.000, 0.000, 1.000, 1.000, 1.000])
    check("Table VI IGRA accs",
          [t6[k]["igra_acc"] for k in
           ["technical_driven", "ethical_driven", "accountability_driven",
            "combined_moderate", "all_low", "all_high"]],
          [1.000, 1.000, 1.000, 0.978, 1.000, 1.000])
    check("Table VI overall", [t6["overall"]["pavi_acc"],
                               t6["overall"]["igra_acc"]], [0.667, 0.996])

    # --- Table VII --------------------------------------------------------
    t7 = sc["table7"]
    check("Table VII dominance", [t7[k]["dominance_acc"] for k in
                                  ["ethical_driven",
                                   "accountability_driven"]], [1.0, 1.0])
    check("Table VII naive-avg", [t7[k]["naive_avg_acc"] for k in
                                  ["ethical_driven",
                                   "accountability_driven"]], [0.0, 0.0])

    # --- Section X-C pure-max comparisons --------------------------------
    check("pure max identical on clean scenarios",
          all(v["identical_to_igra"]
              for v in sc["pure_max_clean"].values()), True)
    check("pure max identical on borderline (81.4% vs 81.4%)",
          b["pure_max_borderline"]["identical_to_igra"], True)
    mb = b["mean_branch"]
    check("mean-branch tier disagreement", mb["tier_disagreement_rate"],
          0.521)
    check("mean-branch IGRA always continue", mb["igra_always_continue"],
          True)
    check("mean-branch accuracies", [mb["igra_acc"], mb["pmax_acc"]],
          [1.0, 1.0])

    # --- Table VIII -------------------------------------------------------
    t8 = b["table8"]
    check("Table VIII PAVI",
          [t8["pavi"]["accuracy"], t8["pavi"]["false_escalation"],
           t8["pavi"]["missed_critical"]], [0.247, 0.000, 1.000])
    check("Table VIII IGRA",
          [t8["igra"]["accuracy"], t8["igra"]["false_escalation"],
           t8["igra"]["missed_critical"]], [0.814, 0.607, 0.048])
    check("Table VIII IGRA balanced accuracy",
          t8["igra"]["balanced_accuracy"], 0.672)

    # --- Section X-D threshold sweep -------------------------------------
    sweep = {r["tau"]: r for r in b["threshold_sweep"]}
    check("sweep flat at tau<=0.35 (acc)",
          [sweep[x]["accuracy"] for x in [0.05, 0.20, 0.35]],
          [0.814, 0.814, 0.814])
    check("sweep balanced-accuracy peak at tau=0.50",
          sweep[0.50]["balanced_accuracy"], 0.755)
    check("sweep false-escalation at tau=0.60",
          sweep[0.60]["false_escalation"], 0.032)
    check("sweep missed-critical at tau=0.60",
          sweep[0.60]["missed_critical"], 0.687)

    # --- Table IX ---------------------------------------------------------
    t9 = b["table9"]
    check("Table IX PAVI accs",
          [t9[k]["pavi_acc"] for k in
           ["max_original", "weighted_sum", "lexicographic",
            "cost_sensitive"]], [0.247, 1.000, 0.453, 0.272])
    check("Table IX IGRA accs",
          [t9[k]["igra_acc"] for k in
           ["max_original", "weighted_sum", "lexicographic",
            "cost_sensitive"]], [0.814, 0.133, 0.656, 0.765])

    # --- Table X ----------------------------------------------------------
    t10 = w["table10"]
    check("Table X IGRA precision",
          [t10[p]["igra"]["precision"] for p in
           ["0.01", "0.05", "0.1", "0.25", "0.5"]],
          [0.333, 0.815, 0.871, 0.956, 0.986])
    check("Table X IGRA recall",
          [t10[p]["igra"]["recall"] for p in
           ["0.01", "0.05", "0.1", "0.25", "0.5"]],
          [0.955, 0.975, 0.981, 0.971, 0.973])
    check("Table X IGRA reviews/1000",
          [t10[p]["igra"]["reviews_per_1000"] for p in
           ["0.01", "0.05", "0.1", "0.25", "0.5"]],
          [25.2, 58.4, 108.8, 245.6, 488.0], nd=1)
    check("Table X PAVI all zero",
          all(t10[p]["pavi"]["precision"] == 0.0
              and t10[p]["pavi"]["recall"] == 0.0
              for p in t10), True)

    # --- Table XI ---------------------------------------------------------
    t11 = fi["table11"]
    check("Table XI ethical_risk",
          [t11[k]["ethical_risk"] for k in
           ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [0.215, 0.140, 0.261, 0.270, 0.402])
    check("Table XI IGRA escalation (%)",
          [t11[k]["igra_escalation"] * 100 for k in
           ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [19.6, 20.0, 18.8, 22.8, 100.0], nd=1)
    check("Table XI PAVI escalation (%)",
          [t11[k]["pavi_escalation"] * 100 for k in
           ["0.0", "0.25", "0.5", "0.75", "1.0"]],
          [19.6, 20.0, 18.8, 22.8, 17.2], nd=1)

    # --- Revision additions: clean-scenario error counts -----------------
    ec = sc["error_counts"]
    check("clean missed-critical, PAVI-only (count)",
          [ec["overall"]["pavi"]["missed"], ec["overall"]["pavi"]["critical"]],
          [1000, 2000])
    check("clean missed-critical, IGRA (count)",
          [ec["overall"]["igra"]["missed"], ec["overall"]["igra"]["critical"]],
          [0, 2000])
    check("clean false escalations, IGRA (count of non-critical)",
          [ec["overall"]["igra"]["false_esc"],
           ec["overall"]["igra"]["noncritical"]], [11, 1000])
    d = ec["combined_moderate"]["false_esc_detail"]
    check("combined_moderate false escalations (all non-critical)", d["n"], 11)
    check("combined_moderate false-esc true max severity (max)",
          d["true_max_severity_max"], 0.349)
    check("combined_moderate false-esc observed max (min)",
          d["observed_max_min"], 0.386)

    # --- Revision additions: delta_E sensitivity --------------------------
    sw = {round(r["delta_E"], 2): r for r in fi["delta_E_sweep"]}
    check("delta_E sweep: escalation at 0.10 / 0.11 (%)",
          [sw[0.10]["igra_escalation_at_mean_eod"] * 100,
           sw[0.11]["igra_escalation_at_mean_eod"] * 100],
          [100.0, 17.2], nd=1)
    check("delta_E sweep: ethical_risk at 0.11", sw[0.11]["ethical_risk"],
          0.366)
    check("delta_E sweep: seed crossings at 0.06,0.08,0.12,0.15,0.20",
          [sw[k]["seed_level_crossings"] for k in
           (0.06, 0.08, 0.12, 0.15, 0.20)], [5, 4, 3, 1, 1])
    check("delta_E sweep: escalation flips only between 0.10 and 0.11",
          [k for k in sorted(sw) if sw[k]["escalates_at_mean_eod"]][-1],
          0.10)

    print(f"\n{PASS} checks passed, {FAIL} failed.")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
