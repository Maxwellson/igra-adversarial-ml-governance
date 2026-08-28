# IGRA — reproducibility package

Code, data, and reproduction scripts for the article:

> **Resilient and Responsible AI Systems: An Integrated Framework for
> Technical, Ethical, and Accountability Governance in Adversarial
> Machine Learning** — Joshua Enoch Maxwellson, Nuhamin Henok Tesfay,
> Elijah Afolabi, Dwayne Livermore, and Alex V. Mbaziira

This repository contains the reference implementation of IGRA (Integrated
Governance Response Algorithm, Algorithm 1 of the paper), the synthetic
fairness dataset generator, the WDBC attack code (FGSM, PGD, label-flip
poisoning), and the scripts that produce every experimental table and
results figure in the article (Tables IV–XI, Figs. 4–6).

## Repository structure

```
src/igra.py                     IGRA core: fusion rule, response bands,
                                decision policies (Algorithm 1)
attacks/attacks.py              FGSM, 20-step PGD (3 restarts), label-flip
                                poisoning
data_generation/
  generate_synthetic_fairness.py  Section IV-A generator (seed 42 CSV)
experiments/
  common.py                     Model architecture and training (Sec. IX-A)
  exp_technical_wdbc.py         Table IV: PAVI under FGSM/PGD/poisoning
  exp_fairness.py               Table V: fairness-targeted poisoning
  exp_scenarios.py              Tables VI–VII: clean scenarios, ablation
  exp_borderline.py             Tables VIII–IX, Fig. 4 data: borderline
                                scenario, threshold sweep, policy oracles,
                                mean-branch comparison
  exp_workload.py               Table X: review workload by prevalence
  exp_fairness_integration.py   Table XI: measured EOD fed into IGRA
  make_figures.py               Figs. 4–6
configs/defaults.json           Every constant from Table II in one place
data/                           Seed-42 synthetic fairness dataset + docs
results/raw/                    JSON outputs of the committed runs
reproduce/
  reproduce_all.py              Runs everything, in order
  verify_reported_results.py    Asserts the paper's reported values
docs/reproducibility_notes.md   Determinism and convention notes
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python reproduce/reproduce_all.py          # ~10-20 min on a modern CPU
python reproduce/verify_reported_results.py
```

`verify_reported_results.py` checks 43 quantities against the values
reported in the paper — every Table IV–XI cell class, the worked trace of
Section VIII-F, the seed-level fairness crossings (4/8), and the
structural equivalences of Section X-C. All 43 pass on the committed
environment (see `requirements.txt`).

## Determinism

All experiments are seeded and single-threaded (`torch.set_num_threads(1)`,
CPU only). The simulation experiments (Tables VI–XI other than the
training-based columns) depend only on NumPy and reproduce exactly across
platforms. The training-based results (Tables IV and V) are exact under
the pinned PyTorch version; other PyTorch builds may shift third-decimal
digits. See `docs/reproducibility_notes.md` for details, including which
implementation conventions the numbers depend on.

## Data

`data/synthetic_fairness_dataset.csv` is the seed-42 instance of the
controlled synthetic fairness dataset (4000 rows; Section IV-A). The
experiments regenerate all eight seed instances from
`data_generation/generate_synthetic_fairness.py`; the CSV is provided so
the exact data can also be archived independently (e.g., Mendeley Data)
under CC BY 4.0. See `data/README.md` and `data/data_dictionary.csv`.

## License

Code: MIT. Dataset: CC BY 4.0. See `LICENSE`.

## Citing

See `CITATION.cff`.

After the first tagged release, add the Zenodo DOI badge here.
