"""Summarize exp_proxy_selection.py: paired changes in the signed equal-opportunity gap
by selection rule, proxy strength, flip fraction and decision threshold."""
import json, os, glob
import numpy as np
from scipy import stats

def main():
    RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "raw")
    DS = [0.0, 0.5, 1.0, 2.0]
    out = {}
    def ci(v):
        v = np.asarray(v); m = v.mean(); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return float(m), float(m - h), float(m + h)
    for d in DS:
        r = json.load(open(os.path.join(RAW, f"proxy_selection_d{d}.json")))
        seeds = [str(s) for s in r["seeds"]]
        for rule in r["rules"]:
            for f in ["0.25", "0.5", "0.75", "1.0"]:
                for t in ["0.3", "0.5", "0.7"]:
                    gap = lambda rec: rec[t]["tpr0"] - rec[t]["tpr1"]
                    dg = [gap(r["runs"][s][rule][f]) - gap(r["runs"][s]["clean"]) for s in seeds]
                    da = [r["runs"][s][rule][f][t]["acc"] - r["runs"][s]["clean"][t]["acc"] for s in seeds]
                    out[f"{rule}|{d}|{f}|{t}"] = {"gap": ci(dg), "acc": ci(da), "n": len(seeds)}
    json.dump(out, open(os.path.join(RAW, "proxy_selection_summary.json"), "w"), indent=1)
    def row(rule, t, f):
        return "  ".join("%+.3f" % out[f"{rule}|{d}|{f}|{t}"]["gap"][0] for d in DS)
    print("Change in signed EO gap, threshold 0.5, columns d = 0, 0.5, 1, 2")
    for f in ["0.25", "0.5", "1.0"]:
        print(" flip", f)
        for rule in ["latent", "random", "far", "upper"]:
            print("  %-7s %s" % (rule, row(rule, "0.5", f)))
    print("Latent rule, flip 0.5, by decision threshold")
    for t in ["0.3", "0.5", "0.7"]:
        print("  t=%s  %s" % (t, row("latent", t, "0.5")))
    print("Accuracy change, threshold 0.5, flip 0.5")
    for rule in ["latent", "random", "far", "upper"]:
        print("  %-7s %s" % (rule, "  ".join("%+.3f" % out[f"{rule}|{d}|0.5|0.5"]["acc"][0] for d in DS)))



if __name__ == "__main__":
    main()
