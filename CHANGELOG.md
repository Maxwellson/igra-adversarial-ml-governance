# Changelog

## 1.2.1 (2026-11-18)
- Added `experiments/exp_proxy_selection.py` and `experiments/analysis_proxy_selection.py`
  with outputs `results/raw/proxy_selection_d*.json` and `proxy_selection_summary.json`.
  They repeat the proxy study with three other ways of choosing which group-1
  positives are flipped (random, farthest from the boundary first, nearest to
  the 75th percentile of the latent score) and measure the equal-opportunity gap
  at decision thresholds of 0.3, 0.5 and 0.7.
- `reproduce/reproduce_proxy_study.py` now also runs the selection-rule study.

## 1.2.0 (2026-11-08)
- Added `experiments/exp_proxy_mechanism.py` and `experiments/analysis_proxy_mechanism.py`
  with outputs `results/raw/proxy_mechanism.json` and `proxy_mechanism_summary.json`.
  They rerun the proxy-study models (identical accuracy to the 1.1.0 run) and add
  a proxy ablation at test time, per-model permutation thresholds for the
  equal-opportunity gap, an attacker that ranks examples by the clean model's
  margin, a repeated-measures ANOVA and mixed-model test of the proxy-by-poisoning
  interaction, and Holm-adjusted paired tests.
- `reproduce/reproduce_proxy_study.py` now also runs the mechanism study.

## 1.1.0 (2026-11-04)
- Added `experiments/exp_proxy_fairness.py`, `experiments/analysis_proxy_fairness.py`
  and `experiments/analysis_fairness_paired.py`, with outputs in `results/raw/`
  (`proxy_fairness.json`, `proxy_fairness_summary.json`, `fairness_paired.json`).
- Added `reproduce/reproduce_proxy_study.py` to rerun the proxy study (about four minutes).
- Added a delta_E sweep to `exp_fairness_integration.py` and error counts to `exp_scenarios.py`.
- `reproduce/verify_reported_results.py` now checks 53 quantities (was 43).
- The proxy study was run with PyTorch 2.15.0, NumPy 2.5.3 and SciPy 1.18.1.

## 1.0.0 (2026-08-27)
- First release.
