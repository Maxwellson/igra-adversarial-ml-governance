"""Runs the proxy-feature fairness study (second article) and its analysis.

Takes about ten minutes on a modern CPU. Writes results/raw/proxy_fairness.json
and proxy_fairness_summary.json, and results/raw/fairness_paired.json (the
chance-gap reference), plus proxy_mechanism.json and
proxy_mechanism_summary.json (proxy ablation, permutation thresholds,
model-based attacker) and proxy_selection_d*.json and
proxy_selection_summary.json (other selection rules and decision thresholds).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from experiments import (analysis_fairness_paired,  # noqa: E402
                         analysis_proxy_fairness, analysis_proxy_mechanism,
                         exp_proxy_fairness, exp_proxy_mechanism,
                         exp_proxy_selection, analysis_proxy_selection)


def main():
    exp_proxy_fairness.run()
    analysis_proxy_fairness.main()
    analysis_fairness_paired.main()
    exp_proxy_mechanism.run()
    analysis_proxy_mechanism.main()
    exp_proxy_selection.run(exp_proxy_selection.PROXY_D)
    analysis_proxy_selection.main()


if __name__ == "__main__":
    main()
