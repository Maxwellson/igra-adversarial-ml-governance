"""Runs every experiment in the paper, in order, writing results/raw/*.json
and figures/*.png. Total runtime is roughly 10-20 minutes on a modern CPU
(the WDBC and fairness retrains dominate).

Order matters: exp_fairness_integration consumes the outputs of
exp_technical_wdbc and exp_fairness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from data_generation import generate_synthetic_fairness  # noqa: E402
from experiments import (exp_borderline, exp_fairness,  # noqa: E402
                         exp_fairness_integration, exp_scenarios,
                         exp_technical_wdbc, exp_workload, make_figures)


def main():
    print("=" * 70)
    print("1/7 Writing the seed-42 synthetic fairness dataset CSV")
    print("=" * 70)
    generate_synthetic_fairness.main()

    print("=" * 70)
    print("2/7 Technical-risk measurement on WDBC (Table IV)")
    print("=" * 70)
    exp_technical_wdbc.run()

    print("=" * 70)
    print("3/7 Fairness-targeted poisoning (Table V)")
    print("=" * 70)
    exp_fairness.run()

    print("=" * 70)
    print("4/7 Clean scenarios and fusion ablation (Tables VI-VII)")
    print("=" * 70)
    exp_scenarios.run()

    print("=" * 70)
    print("5/7 Borderline scenario, threshold sweep, policy oracles "
          "(Tables VIII-IX, Fig. 4)")
    print("=" * 70)
    exp_borderline.run()

    print("=" * 70)
    print("6/7 Review workload (Table X)")
    print("=" * 70)
    exp_workload.run()

    print("=" * 70)
    print("7/7 Fairness-signal integration (Table XI) and figures")
    print("=" * 70)
    exp_fairness_integration.run()
    make_figures.main()

    print("\nDone. Now run reproduce/verify_reported_results.py to check "
          "the outputs against the values reported in the paper.")


if __name__ == "__main__":
    main()
