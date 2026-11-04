# Changelog

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
