"""Analysis for the proxy study: factorial test, Holm correction, permutation-null
flags, proxy ablation, model-based attacker, and proxy information content.

Reads results/raw/proxy_fairness.json and results/raw/proxy_mechanism.json.
Writes results/raw/proxy_mechanism_summary.json.
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")

def ci(d):
    n = len(d); se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return [float(d.mean() - tc * se), float(d.mean() + tc * se)]

def main():
    A = json.load(open(os.path.join(RAW, "proxy_fairness.json")))
    M = json.load(open(os.path.join(RAW, "proxy_mechanism.json")))
    seeds = [str(s) for s in A["seeds"]]; ds = [str(d) for d in A["proxy_d"]]
    fr = [str(f) for f in A["fractions"][1:]]
    out = {}
    # reproduction check: mechanism run retrains the same models
    mx = max(abs(A["runs"][d][s][f]["acc"] - M["runs"][d][s]["latent"][f]["acc"])
             for d in ds for s in seeds for f in ["0.0"] + fr)
    out["max_acc_diff_vs_main_run"] = float(mx)

    # ---- long table of seed-level changes
    rows = []
    for d in ds:
        for s in seeds:
            r = A["runs"][d][s]
            e0 = r["0.0"]["tpr0"] - r["0.0"]["tpr1"]
            for f in fr:
                rows.append({"seed": s, "d": d, "f": f,
                             "deod": (r[f]["tpr0"] - r[f]["tpr1"]) - e0,
                             "dacc": r[f]["acc"] - r["0.0"]["acc"]})
    df = pd.DataFrame(rows)
    # ---- factorial analysis
    fac = {}
    for y in ("deod", "dacc"):
        aov = AnovaRM(df, y, "seed", within=["d", "f"]).fit().anova_table
        mm = {}
        for name, form in (("full", f"{y} ~ C(d)*C(f)"), ("noint", f"{y} ~ C(d)+C(f)")):
            mm[name] = smf.mixedlm(form, df, groups=df["seed"]).fit(reml=False)
        lr = 2 * (mm["full"].llf - mm["noint"].llf); dof = 9
        df2 = df.assign(dn=df["d"].astype(float), fn=df["f"].astype(float))
        t1 = smf.mixedlm(f"{y} ~ dn*fn", df2, groups=df2["seed"]).fit(reml=True)
        fac[y] = {"rm_anova": {k: {"F": float(aov.loc[k, "F Value"]), "df": [float(aov.loc[k, "Num DF"]), float(aov.loc[k, "Den DF"])],
                                   "p": float(aov.loc[k, "Pr > F"])} for k in aov.index},
                  "mixed_lrt_interaction": {"chi2": float(lr), "df": dof, "p": float(stats.chi2.sf(lr, dof))},
                  "linear_interaction": {"coef": float(t1.params["dn:fn"]), "z": float(t1.tvalues["dn:fn"]), "p": float(t1.pvalues["dn:fn"])}}
    out["factorial"] = fac
    # ---- Holm across the 16 paired comparisons of delta EOG (and accuracy)
    cells, pt, pw = [], [], []
    for d in ds:
        for f in fr:
            x = df[(df.d == d) & (df.f == f)]["deod"].values
            cells.append((d, f)); pt.append(stats.ttest_1samp(x, 0).pvalue); pw.append(stats.wilcoxon(x).pvalue)
    ht = multipletests(pt, method="holm")[1]; hw = multipletests(pw, method="holm")[1]
    out["holm_deod"] = [{"d": c[0], "f": c[1], "p_t": float(a), "holm_t": float(b), "p_w": float(cc), "holm_w": float(e)}
                        for c, a, b, cc, e in zip(cells, pt, ht, pw, hw)]
    # ---- permutation-null flags
    flags = {}
    for d in ds:
        flags[d] = {}
        for f in ["0.0"] + fr:
            ms = [M["runs"][d][s]["latent"][f] for s in seeds]
            flags[d][f] = {"above_own_p95": int(sum(abs(m["tpr0"] - m["tpr1"]) > m["null95"] for m in ms)),
                           "perm_p_lt_05": int(sum(m["perm_p"] < 0.05 for m in ms)),
                           "mean_null95": float(np.mean([m["null95"] for m in ms])),
                           "min_null95": float(np.min([m["null95"] for m in ms])),
                           "max_null95": float(np.max([m["null95"] for m in ms]))}
    out["perm_flags"] = flags
    # ---- proxy ablation
    ab = {}
    for d in ds[1:]:
        ab[d] = {}
        for f in fr:
            ms = [M["runs"][d][s]["latent"][f] for s in seeds]
            e = np.array([m["tpr0"] - m["tpr1"] for m in ms])  # positive = group 1 disadvantaged
            sh = np.array([m["abl_shuffle_eod"] for m in ms])
            ze = np.array([m["abl_zero_eod"] for m in ms])
            dd = e - sh
            ab[d][f] = {"gap": float(e.mean()), "gap_shuffled": float(sh.mean()), "gap_zeroed": float(ze.mean()),
                        "drop_ci": ci(dd), "share_reduced": float((dd > 0).mean())}
    out["ablation_gap_group1_disadvantage"] = ab
    # ---- model-based attacker vs latent attacker (signed EOD change and accuracy change)
    mb = {}
    for d in ds:
        mb[d] = {}
        for f in fr:
            dl, dm, al, am = [], [], [], []
            for s in seeds:
                c = M["runs"][d][s]["latent"]["0.0"]; e0 = c["tpr0"] - c["tpr1"]
                l = M["runs"][d][s]["latent"][f]; m = M["runs"][d][s]["model"][f]
                dl.append((l["tpr0"] - l["tpr1"]) - e0); dm.append((m["tpr0"] - m["tpr1"]) - e0)
                al.append(l["acc"] - c["acc"]); am.append(m["acc"] - c["acc"])
            dl, dm, al, am = map(np.array, (dl, dm, al, am))
            mb[d][f] = {"latent_deod": float(dl.mean()), "model_deod": float(dm.mean()), "model_deod_ci": ci(dm),
                        "latent_dacc": float(al.mean()), "model_dacc": float(am.mean())}
    out["model_attacker"] = mb
    # ---- proxy information content
    rng = np.random.default_rng(0); n = 1_000_000
    g = (rng.random(n) < 0.35).astype(int); info = {}
    for d in [0.0, 0.5, 1.0, 2.0]:
        p = d * g + rng.normal(0, 1, n)
        auc = float(stats.rankdata(p)[g == 1].sum() - (g == 1).sum() * ((g == 1).sum() + 1) / 2) / ((g == 1).sum() * (g == 0).sum())
        bal = float(0.5 * ((p[g == 1] > d / 2).mean() + (p[g == 0] <= d / 2).mean()))
        info[str(d)] = {"auc": auc, "auc_closed_form": float(stats.norm.cdf(d / np.sqrt(2))), "balanced_acc_midpoint": bal}
    out["proxy_information"] = info
    # ---- attack size relative to the whole training set
    nt = np.mean([M["runs"]["0.0"][s]["n_target"] for s in seeds]); ntr = M["runs"]["0.0"][seeds[0]]["n_train"]
    out["attack_size"] = {"mean_target": float(nt), "n_train": int(ntr), "share_of_training_by_fraction": {f: float(nt * float(f) / ntr) for f in fr}}
    json.dump(out, open(os.path.join(RAW, "proxy_mechanism_summary.json"), "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("max_acc_diff_vs_main_run", "factorial", "proxy_information", "attack_size")}, indent=1))
    print("HOLM", [(x["d"], x["f"], round(x["holm_t"], 4)) for x in out["holm_deod"] if x["holm_t"] > 0.01])
    for d in ds: print("FLAGS", d, {f: (v["above_own_p95"], v["perm_p_lt_05"]) for f, v in flags[d].items()}, "null95 mean", round(flags[d]["0.0"]["mean_null95"], 3), "range", round(min(v["min_null95"] for v in flags[d].values()), 3), round(max(v["max_null95"] for v in flags[d].values()), 3))
    for d in ds[1:]: print("ABL", d, {f: (round(v["gap"], 3), round(v["gap_shuffled"], 3), round(v["gap_zeroed"], 3), v["share_reduced"]) for f, v in ab[d].items()})
    for d in ds: print("MODELATK", d, {f: (round(v["latent_deod"], 3), round(v["model_deod"], 3), round(v["latent_dacc"], 3), round(v["model_dacc"], 3)) for f, v in mb[d].items()})

if __name__ == "__main__":
    main()
