# Reproducibility notes

## Environment

The committed `results/raw/*.json` were produced with the exact versions
in `requirements.txt` (PyTorch 2.13.0, NumPy 2.4.4, scikit-learn 1.8.0)
on CPU with `torch.set_num_threads(1)`. Under this environment the full
pipeline is deterministic end to end, and
`reproduce/verify_reported_results.py` passes all 43 checks.

## What the reported numbers depend on

Three implementation conventions are load-bearing for digit-exact
reproduction; all three are stated in the paper and encoded here:

1. **FGSM is the plain single-step attack; PGD additionally clips to the
   training set's observed per-feature range** (Section IX-A attaches the
   clipping clause to PGD). Clipping FGSM as well lowers its measured
   PAVI at large epsilon by roughly 0.005-0.015.
2. **Table IV and the clean-accuracy spread use the population standard
   deviation** (ddof = 0) across the five seeds, matching the paper's
   96.37% +/- 0.44%.
3. **Random draws are keyed to documented seeds** (`configs/defaults.json`).
   The poisoning flip indices use a per-(seed, fraction) derived seed
   (`seed*100 + int(fraction*100)`); the PGD restart initialization uses
   `seed*1000 + int(eps*100)`; the synthetic fairness generator consumes
   its RNG in the order g, feature_1, feature_2, noise block, label noise.
   Changing any of these draw orders changes trailing digits (and, for
   the poisoning column, the realized per-cell values) without changing
   any qualitative conclusion.

## Cross-version expectations

- Tables VI-XI's simulation components are NumPy-only and reproduce
  exactly on any platform with the pinned NumPy.
- Tables IV-V involve PyTorch training. On other PyTorch builds, expect
  identical qualitative patterns with possible third-decimal shifts;
  the verification script will then report which cells moved.

## Relationship to the paper

The paper's Tables IV-XI report exactly the outputs of this code under
the pinned environment and the seeds in `configs/defaults.json` - the
verification script is the machine-checkable statement of that claim.
