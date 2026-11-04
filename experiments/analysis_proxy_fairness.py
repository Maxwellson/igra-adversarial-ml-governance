"""Paired analysis of results/raw/proxy_fairness.json.

For each proxy strength d and poisoning fraction f, reports paired seed-level
differences from the same seed's clean model (f = 0) for accuracy, the SIGNED
equal-opportunity gap (TPR of group 0 minus TPR of group 1), the signed
selection-rate gap, and group-1 TPR, each with a 95% t interval, a paired t
test and a Wilcoxon signed-rank test. Also reports the mean absolute EOD (the
folded metric of the original study) and the clean-model signed gap, which
should be near zero. Writes results/raw/proxy_fairness_summary.json.
"""
import json, os
import numpy as np
from scipy import stats

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")

def paired(d):
    n = len(d); sd = d.std(ddof=1); se = sd / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_1samp(d, 0.0)
    try: pw = float(stats.wilcoxon(d).pvalue)
    except ValueError: pw = float("nan")
    return {"mean": float(d.mean()), "ci": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
            "p_t": float(p), "p_w": pw, "n_pos": int((d > 0).sum()), "n": n,
            "dz": float(d.mean() / sd) if sd > 0 else float("nan")}

def main():
    r = json.load(open(os.path.join(RAW, "proxy_fairness.json")))
    seeds = [str(s) for s in r["seeds"]]; fracs = [str(f) for f in r["fractions"]]
    out = {"n_seeds": len(seeds), "by_d": {}}
    for d, runs in r["runs"].items():
        def col(f, key): return np.array([runs[s][f][key] for s in seeds])
        def eod(f): return col(f, "tpr0") - col(f, "tpr1")
        def dpd(f): return col(f, "sel0") - col(f, "sel1")
        res = {"clean_signed_eod": paired(eod("0.0")), "clean_abs_eod_mean": float(np.abs(eod("0.0")).mean()),
               "clean_acc": float(col("0.0", "acc").mean()), "frac": {}}
        for f in fracs[1:]:
            res["frac"][f] = {
                "acc": paired(col(f, "acc") - col("0.0", "acc")),
                "signed_eod": paired(eod(f) - eod("0.0")),
                "signed_dpd": paired(dpd(f) - dpd("0.0")),
                "tpr1": paired(col(f, "tpr1") - col("0.0", "tpr1")),
                "tpr0": paired(col(f, "tpr0") - col("0.0", "tpr0")),
                "abs_eod_mean": float(np.abs(eod(f)).mean()),
                "signed_eod_mean": float(eod(f).mean()),
                "signed_eod_sd": float(eod(f).std(ddof=1)),
                "acc_mean": float(col(f, "acc").mean())}
        out["by_d"][d] = res
    # Share of models whose absolute signed gap reaches the 95th percentile of a fair
    # model's gap on a 1,200-case test split at true-positive rate 0.8 (simulated in
    # analysis_fairness_paired.py).
    thr = json.load(open(os.path.join(RAW, "fairness_paired.json")))["noise_floor"]["0.8"]["p95"]
    out["flag_threshold"] = thr
    for d, runs in r["runs"].items():
        out["by_d"][d]["flag_share"] = {
            f: float(np.mean([abs(runs[s][f]["tpr0"] - runs[s][f]["tpr1"]) >= thr for s in seeds]))
            for f in fracs}
        out["by_d"][d]["clean_acc_sd"] = float(np.std([runs[s]["0.0"]["acc"] for s in seeds], ddof=1))
    json.dump(out, open(os.path.join(RAW, "proxy_fairness_summary.json"), "w"), indent=2)
    for d, res in out["by_d"].items():
        c = res["clean_signed_eod"]
        print(f"\n== proxy d={d}  clean acc {res['clean_acc']:.3f}  clean signed EOD {c['mean']:+.4f} "
              f"[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]  clean |EOD| {res['clean_abs_eod_mean']:.4f}")
        for f, v in res["frac"].items():
            e, a, t1 = v["signed_eod"], v["acc"], v["tpr1"]
            print(f" f={f}: dAcc {a['mean']:+.3f} | dEOD {e['mean']:+.4f} [{e['ci'][0]:+.4f},{e['ci'][1]:+.4f}] "
                  f"p={e['p_t']:.4f} W={e['p_w']:.4f} up={e['n_pos']}/{e['n']} dz={e['dz']:.2f} | "
                  f"dTPR1 {t1['mean']:+.3f} | signedEOD {v['signed_eod_mean']:+.3f} sd {v['signed_eod_sd']:.3f}")

if __name__ == "__main__":
    main()
