"""Paired analysis of the fairness-poisoning results (companion fairness paper).

Reads results/raw/fairness.json (8 seeds x 5 poisoning fractions) and reports
paired seed-level statistics: EOD/DPD/accuracy change from clean to each
poisoned fraction, paired t and Wilcoxon tests, and the across-seed
correlation between accuracy loss and EOD at full strength.
Writes results/raw/fairness_paired.json.
"""
import json, os
import numpy as np
from scipy import stats

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")

def main():
    f = json.load(open(os.path.join(RAW, "fairness.json")))
    seeds = [str(s) for s in f["seeds"]]
    fracs = [str(x) for x in f["fractions"]]
    out = {"n_seeds": len(seeds), "paired": {}}
    def col(frac, key):
        return np.array([f["per_seed"][s][frac][key] for s in seeds])
    for frac in fracs[1:]:
        row = {}
        for key in ("eod", "dpd", "acc"):
            d = col(frac, key) - col("0.0", key)
            n = len(d); sd = d.std(ddof=1); se = sd / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            t, p = stats.ttest_1samp(d, 0.0)
            try:
                w = stats.wilcoxon(d)
                wp = float(w.pvalue)
            except ValueError:
                wp = float("nan")
            row[key] = {"mean_diff": float(d.mean()), "ci": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
                        "t": float(t), "p_t": float(p), "p_wilcoxon": wp,
                        "n_positive": int((d > 0).sum()), "dz": float(d.mean() / sd) if sd > 0 else float("nan")}
        out["paired"][frac] = row
    acc_loss = col("0.0", "acc") - col("1.0", "acc")
    eod = col("1.0", "eod")
    r, p = stats.pearsonr(acc_loss, eod)
    rho, ps = stats.spearmanr(acc_loss, eod)
    out["accloss_vs_eod_full"] = {"pearson_r": float(r), "p": float(p), "spearman_rho": float(rho), "p_spearman": float(ps)}
    out["clean_eod_per_seed"] = col("0.0", "eod").tolist()
    out["full_eod_per_seed"] = eod.tolist()
    # EOD alarm rule: escalate when context_weight * min(1, EOD/delta_E) >= 0.50,
    # i.e. when EOD >= theta = (0.50 / 1.30) * delta_E.
    clean = np.array(out["clean_eod_per_seed"]); full = np.array(out["full_eod_per_seed"])
    ratio = 0.50 / 1.30
    out["alarm_sweep"] = []
    for k in range(15):
        dE = round(0.06 + 0.01 * k, 2); theta = ratio * dE
        out["alarm_sweep"].append({
            "delta_E": dE, "theta": theta,
            "mean_eod_full_alarm": bool(f["table5"]["1.0"]["eod"] >= theta),
            "mean_eod_clean_alarm": bool(f["table5"]["0.0"]["eod"] >= theta),
            "alarms_clean_seeds": int((clean >= theta).sum()),
            "alarms_full_seeds": int((full >= theta).sum())})
    u = stats.mannwhitneyu(full, clean, alternative="greater")
    out["auc_full_vs_clean"] = {"auc": float(u.statistic / (len(full) * len(clean))),
                                "p_one_sided": float(u.pvalue)}
    out["seeds_needed_80pct_power"] = None
    d = full - clean; dz = d.mean() / d.std(ddof=1)
    for n in range(4, 200):
        tc = stats.t.ppf(0.975, n - 1); nc = dz * np.sqrt(n)
        if 1 - stats.nct.cdf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc) >= 0.80:
            out["seeds_needed_80pct_power"] = n; break
    out["paired_eod_full_sd_diff"] = float(d.std(ddof=1))
    # Sampling-noise floor of |EOD|. g is independent of (x, y) and is not a model
    # input, so the population EOD of every trained model is zero; the measured
    # |EOD| on a 1200-example test split is then pure finite-sample noise.
    rng = np.random.default_rng(0)
    floor = {}
    n_test, p_g = 1200, 0.35
    for tpr in (0.5, 0.6, 0.7, 0.8, 0.9):
        vals = []
        for _ in range(20000):
            g = rng.random(n_test) < p_g
            y = np.zeros(n_test, bool); y[rng.permutation(n_test)[: n_test // 2]] = True
            pred = rng.random(n_test) < tpr
            a, b = y & g, y & ~g
            vals.append(abs(pred[a].mean() - pred[b].mean()))
        floor[str(tpr)] = {"mean_abs_eod": float(np.mean(vals)),
                           "p95": float(np.percentile(vals, 95))}
    out["noise_floor"] = floor
    json.dump(out, open(os.path.join(RAW, "fairness_paired.json"), "w"), indent=2)
    for frac, row in out["paired"].items():
        e = row["eod"]
        print(f"frac {frac}: dEOD {e['mean_diff']:+.4f} CI [{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] "
              f"t p={e['p_t']:.3f} W p={e['p_wilcoxon']:.3f} up={e['n_positive']}/8 | "
              f"dDPD {row['dpd']['mean_diff']:+.4f} p={row['dpd']['p_t']:.3f} | dAcc {row['acc']['mean_diff']:+.4f} p={row['acc']['p_t']:.3f}")
    print("acc loss vs EOD (full):", out["accloss_vs_eod_full"])
    for r in out["alarm_sweep"]:
        print(f"dE {r['delta_E']:.2f} theta {r['theta']:.4f} mean-full {r['mean_eod_full_alarm']} "
              f"clean seeds {r['alarms_clean_seeds']}/8 full seeds {r['alarms_full_seeds']}/8")
    for k, v in out["noise_floor"].items():
        print(f"noise floor TPR {k}: mean |EOD| {v['mean_abs_eod']:.4f}, 95th pct {v['p95']:.4f}")
    print("AUC", out["auc_full_vs_clean"], "seeds for 80% power", out["seeds_needed_80pct_power"],
          "sd diff", out["paired_eod_full_sd_diff"])
    print("clean EOD per seed", np.round(out["clean_eod_per_seed"],3))
    print("full  EOD per seed", np.round(out["full_eod_per_seed"],3))

if __name__ == "__main__":
    main()
