# Controlled synthetic fairness dataset

Seed-42 instance of the dataset specified in Section IV-A of the paper
(4000 rows, 11 columns). The protected-group attribute is generated
independently of both the predictive features and the label-generating
process, so baseline group parity holds by design and any post-attack
disparity is attributable to the induced poisoning intervention.

- Rows: 4000. Label split: 50/50 by construction (median threshold).
- Protected group: ~65/35 split, drawn independently of everything else.
- The protected attribute and latent logit are included for auditability
  and for reproducing the boundary-targeted attack; only feature_1,
  feature_2, and the six noise features are model inputs.
- Regenerate any seed with data_generation/generate_synthetic_fairness.py.

See data_dictionary.csv for the column-level description.
License: CC BY 4.0 (see repository LICENSE).
