# IGRA — Integrated Governance Response Algorithm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22141626.svg)](https://doi.org/10.5281/zenodo.22141626)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducibility package for:

> **Resilient and Responsible AI Systems: An Integrated Framework for
> Technical, Ethical, and Accountability Governance in Adversarial Machine
> Learning** — Joshua Enoch Maxwellson, Nuhamin Henok Tesfay, Elijah
> Afolabi, Dwayne Livermore, and Alex V. Mbaziira.

---

## The problem

When an adversarial attack degrades a deployed model, the usual response is
to measure how much accuracy survived. That misses two things a single
accuracy number cannot express.

A poisoning attack can shift a model's error rate against one demographic
subgroup while leaving aggregate accuracy nearly unchanged — a fairness
failure invisible to an accuracy-based monitor. Separately, an incident can
be logged so incompletely that it cannot be reconstructed afterwards — an
accountability failure with no accuracy signature at all.

An organisation watching only the technical channel will not escalate
either one.

## What IGRA does

IGRA takes three risk channels and returns one response action.

The rule is **non-compensatory**. If any channel is severe, the fused risk
is the *maximum*, not the mean — so a serious ethical problem is not
cancelled out by clean technical and accountability signals.

```python
from src.igra import igra_decide

# only the ethical channel is alarming
action, record = igra_decide(technical_risk=0.02,
                             ethical_risk=0.70,
                             accountability_risk=0.03,
                             context_weight=1.30)

print(action)                      # fail_safe
print(record['governance_risk'])   # 0.91
```

Averaging the same three values gives 0.25, weights to 0.325, and returns
`continue` — the incident is never escalated. That gap is the design
argument, and `exp_scenarios.py` measures it.

Fused risk maps to one of six response tiers, from `continue` through
`human_review` to `fail_safe`. Organisations below governance maturity
level 3 have automated tiers rerouted to human review. Everything is
specified in `src/igra.py`, which is short and is the file to read first.

## Headline results

| | PAVI-only | IGRA |
|---|---|---|
| Clean scenarios, accuracy | 0.667 | 0.996 |
| Borderline scenarios, accuracy | 0.247 | 0.814 |
| Borderline, missed critical | 1.000 | 0.048 |
| Borderline, false escalation | 0.000 | 0.607 |

The advantage is bought, not free: IGRA misses far fewer critical incidents
but escalates far more non-critical ones. It also **loses** against a
compensatory policy oracle (0.133 against 1.000, Table IX), which is
reported in the paper because it bounds when the approach applies.

## Quick start

```bash
git clone https://github.com/Maxwellson/igra-adversarial-ml-governance.git
cd igra-adversarial-ml-governance
python reproduce/reproduce_all.py          # about 90 seconds
python reproduce/verify_reported_results.py
```

No installation step is needed beyond NumPy, PyTorch, scikit-learn, SciPy
and Matplotlib. `requirements.txt` pins the exact versions behind the
published numbers; installing those pins is only necessary for digit-exact
reproduction.

Runtime by experiment:

| Script | Time |
|---|---|
| `exp_scenarios.py` | 0.4 s |
| `exp_borderline.py` | 0.15 s |
| `exp_workload.py` | 0.12 s |
| `exp_fairness_integration.py` | 0.11 s |
| `exp_technical_wdbc.py` | 8.6 s |
| `exp_fairness.py` | 81 s |

## Verification

`reproduce/verify_reported_results.py` checks **53 quantities** against the
values printed in the article: every Table IV–XI cell class, the worked
trace of Section VIII-F, the 4-of-8 seed-level fairness crossings, the clean-scenario
missed-critical and false-escalation counts, the δ_E sensitivity sweep, and
the structural equivalences of Section X-C. All 53 pass in the pinned
environment.

On other PyTorch builds expect roughly 41 of 53, and slightly fewer if the
δ_E sweep moves with the retrained fairness models. The checks that move are
in Tables IV, V and XI and the δ_E sweep, which all involve a retrained
model; the largest deviation observed is 0.004. The simulation-only checks
(including the six new clean-scenario counts) are NumPy-only and reproduce
exactly on any platform. See
`docs/reproducibility_notes.md`.

## Repository structure

```
src/igra.py                       IGRA core: fusion rule, response bands,
                                  decision policies (Algorithm 1)
attacks/attacks.py                FGSM, 20-step PGD (3 restarts),
                                  label-flip poisoning
data_generation/
  generate_synthetic_fairness.py  Section IV-A generator (seed 42 CSV)
experiments/
  common.py                       Model architecture and training (IX-A)
  exp_technical_wdbc.py           Table IV: PAVI under FGSM/PGD/poisoning
  exp_fairness.py                 Table V: fairness-targeted poisoning
  exp_scenarios.py                Tables VI–VII: clean scenarios, ablation
  exp_borderline.py               Tables VIII–IX, Fig. 4: borderline
                                  scenario, threshold sweep, policy oracles
  exp_workload.py                 Table X: review workload by prevalence
  exp_fairness_integration.py     Table XI: measured EOD fed into IGRA
  make_figures.py                 Figs. 4–6
configs/defaults.json             Every constant from Table II
data/                             Seed-42 dataset, dictionary, README
results/raw/                      JSON outputs of the committed runs
reproduce/
  reproduce_all.py                Runs everything, in order
  verify_reported_results.py      Asserts the reported values
docs/reproducibility_notes.md     Determinism and convention notes
```

## Data

`data/synthetic_fairness_dataset.csv` is the seed-42 instance of the
controlled synthetic fairness dataset (4000 rows, Section IV-A). The
protected attribute is generated independently of both the predictive
features and the label-generating process, so baseline group parity holds
by construction and any post-attack disparity is attributable to the
induced poisoning.

The experiments regenerate all eight seed instances from the generator
script. The CSV is also archived separately at Mendeley Data,
[10.17632/r94kjvbycw.1](https://doi.org/10.17632/r94kjvbycw.1). See
`data/README.md` and `data/data_dictionary.csv`.

## Citing

See `CITATION.cff`, or cite the archived release:

> J. E. Maxwellson, N. H. Tesfay, E. Afolabi, D. Livermore and
> A. V. Mbaziira, *IGRA: reproducibility package*, v1.0.0, Zenodo, 2026.
> doi: 10.5281/zenodo.22141626

## License

Code: MIT. Dataset: CC BY 4.0. See `LICENSE`.
