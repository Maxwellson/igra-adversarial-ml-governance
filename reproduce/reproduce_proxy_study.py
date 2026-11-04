"""Runs the proxy-feature fairness study (second article) and its analysis.

Takes about four minutes on a modern CPU. Writes results/raw/proxy_fairness.json
and proxy_fairness_summary.json, and results/raw/fairness_paired.json (the
chance-gap reference).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from experiments import (analysis_fairness_paired,  # noqa: E402
                         analysis_proxy_fairness, exp_proxy_fairness)


def main():
    exp_proxy_fairness.run()
    analysis_proxy_fairness.main()
    analysis_fairness_paired.main()


if __name__ == "__main__":
    main()
