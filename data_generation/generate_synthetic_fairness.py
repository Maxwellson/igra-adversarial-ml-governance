"""Controlled synthetic fairness dataset generator (Section IV-A).

The generator draws N = 4000 instances with NumPy's default_rng.
Protected group g in {0, 1} is drawn independently with p = [0.65, 0.35].
Two informative features are drawn independently of g:
    feature_1 ~ Normal(0.5, 0.45^2)
    feature_2 ~ Normal(0.45, 0.45^2)
Six noise features are i.i.d. Normal(0, 1), independent of everything else.
The binary label is y = 1{logit > median(logit)} with
    logit = 1.2 * feature_1 + 0.8 * feature_2 + Normal(0, 0.5^2),
a deliberately group-independent process (g does not appear), giving a
50/50 label split by construction via the median threshold. g is NOT a
model input; only feature_1, feature_2, and the six noise features are
passed to the classifier.

Run as a script to write data/synthetic_fairness_dataset.csv (seed 42).
"""

from __future__ import annotations

import os

import numpy as np

N_INSTANCES = 4000
N_NOISE = 6
GROUP_P = [0.65, 0.35]
COEF_1 = 1.2
COEF_2 = 0.8
LABEL_NOISE_SD = 0.5
FEATURE_SD = 0.45


def make_dataset(seed: int):
    """Returns (X, y, g, logit, median_logit).

    X has 8 columns: feature_1, feature_2, noise_1..noise_6.
    The draw order below (g, feature_1, feature_2, noise, label noise) is
    part of the specification: changing it changes the realized dataset
    for a given seed.
    """
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 1], size=N_INSTANCES, p=GROUP_P)
    f1 = rng.normal(0.5, FEATURE_SD, N_INSTANCES)
    f2 = rng.normal(0.45, FEATURE_SD, N_INSTANCES)
    noise = rng.normal(0.0, 1.0, size=(N_INSTANCES, N_NOISE))
    label_noise = rng.normal(0.0, LABEL_NOISE_SD, N_INSTANCES)

    logit = COEF_1 * f1 + COEF_2 * f2 + label_noise
    median_logit = float(np.median(logit))
    y = (logit > median_logit).astype(np.int64)

    x = np.column_stack([f1, f2, noise])
    return x, y, g, logit, median_logit


def main(out_path=None, seed: int = 42):
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "..", "data",
                                "synthetic_fairness_dataset.csv")
    x, y, g, logit, _ = make_dataset(seed)
    header = ("feature_1,feature_2,noise_1,noise_2,noise_3,noise_4,"
              "noise_5,noise_6,protected_group,label,latent_logit")
    rows = np.column_stack([x, g, y, logit])
    np.savetxt(out_path, rows, delimiter=",", header=header, comments="",
               fmt=["%.10f"] * 8 + ["%d", "%d", "%.10f"])
    print(f"Wrote {out_path} (seed={seed}, N={len(y)}, "
          f"positive rate={y.mean():.3f}, group-1 rate={g.mean():.3f})")


if __name__ == "__main__":
    main()
