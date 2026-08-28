"""IGRA: Integrated Governance Response Algorithm.

Reference implementation of Algorithm 1 in the paper
"Resilient and Responsible AI Systems: An Integrated Framework for
Technical, Ethical, and Accountability Governance in Adversarial
Machine Learning."

All defaults match Table II of the paper.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Design defaults (Table II)
# ---------------------------------------------------------------------------
TAU_TECH = 0.20
TAU_ETHICS = 0.20
TAU_ACCOUNT = 0.20

# Response bands (Section VIII-C): upper bound (exclusive) -> action
RESPONSE_BANDS = [
    (0.20, "continue"),
    (0.35, "increase_monitoring"),
    (0.50, "technical_mitigation"),
    (0.70, "human_review"),
    (0.85, "suspend_automation"),
    (float("inf"), "fail_safe"),
]

ESCALATING_ACTIONS = {"human_review", "suspend_automation", "fail_safe"}

# Predicted-critical boundary implied by the response mapping: an action is
# predicted-critical iff governance_risk >= the human_review lower bound.
HUMAN_REVIEW_BOUNDARY = 0.50

# context_weight lookup (Table III)
CONTEXT_WEIGHT_TABLE = {
    ("low", False): 1.00,
    ("low", True): 1.15,
    ("high", False): 1.30,   # used in Sections IX-XI
    ("high", True): 1.50,
}


def fuse(components, weights=None):
    """Equal-weighted fusion with renormalization over available signals.

    ``components`` is an iterable in which missing signals are ``None``
    (Section VIII-D: weights renormalize across available signals; a
    missing signal is not treated as zero).
    """
    vals = [c for c in components if c is not None]
    if not vals:
        raise ValueError("fuse() requires at least one available signal")
    if weights is None:
        return float(np.mean(vals))
    w = [wt for c, wt in zip(components, weights) if c is not None]
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return float(np.dot(w, vals))


def select_response(governance_risk: float) -> str:
    for upper, action in RESPONSE_BANDS:
        if governance_risk < upper:
            return action
    return "fail_safe"


def igra_decide(technical_risk, ethical_risk, accountability_risk,
                context_weight=1.30,
                tau_tech=TAU_TECH, tau_ethics=TAU_ETHICS,
                tau_account=TAU_ACCOUNT, maturity_level=5):
    """Scalar version of Algorithm 1. Returns (action, audit_record)."""
    technical_risk = float(np.clip(technical_risk, 0.0, 1.0))
    ethical_risk = float(np.clip(ethical_risk, 0.0, 1.0))
    accountability_risk = float(np.clip(accountability_risk, 0.0, 1.0))

    dominant = (technical_risk > tau_tech
                or ethical_risk > tau_ethics
                or accountability_risk > tau_account)
    if dominant:
        fused = max(technical_risk, ethical_risk, accountability_risk)
    else:
        fused = (technical_risk + ethical_risk + accountability_risk) / 3.0

    governance_risk = min(1.0, context_weight * fused)
    action = select_response(governance_risk)

    # Maturity floor (Section VIII-C): below Level 3, unattended automated
    # responses are rerouted to human review; fail_safe is never suppressed.
    if maturity_level < 3 and action in {"technical_mitigation",
                                         "suspend_automation"}:
        action = "human_review"

    audit_record = {
        "technical_risk": technical_risk,
        "ethical_risk": ethical_risk,
        "accountability_risk": accountability_risk,
        "dominant_branch": bool(dominant),
        "governance_risk": governance_risk,
        "action": action,
    }
    return action, audit_record


# ---------------------------------------------------------------------------
# Vectorized policies used by the simulation experiments. Each returns a
# boolean array: True where the policy escalates past human review, i.e.
# governance_risk >= 0.50 (Section IX-C's predicted-critical mapping).
# ---------------------------------------------------------------------------

def _governance_dominance(tech, eth, acc, cw, tau):
    tech = np.clip(tech, 0.0, 1.0)
    eth = np.clip(eth, 0.0, 1.0)
    acc = np.clip(acc, 0.0, 1.0)
    stacked = np.stack([tech, eth, acc])
    dominant = (tech > tau) | (eth > tau) | (acc > tau)
    fused = np.where(dominant, stacked.max(axis=0), stacked.mean(axis=0))
    return np.minimum(1.0, cw * fused)


def igra_escalates(tech, eth, acc, cw=1.30, tau=TAU_TECH):
    return _governance_dominance(tech, eth, acc, cw, tau) >= HUMAN_REVIEW_BOUNDARY


def igra_governance(tech, eth, acc, cw=1.30, tau=TAU_TECH):
    return _governance_dominance(tech, eth, acc, cw, tau)


def pavi_only_escalates(tech, cw=1.30):
    return np.minimum(1.0, cw * np.clip(tech, 0.0, 1.0)) >= HUMAN_REVIEW_BOUNDARY


def pure_max_escalates(tech, eth, acc, cw=1.30):
    m = np.maximum.reduce([np.clip(tech, 0, 1), np.clip(eth, 0, 1),
                           np.clip(acc, 0, 1)])
    return np.minimum(1.0, cw * m) >= HUMAN_REVIEW_BOUNDARY


def pure_max_governance(tech, eth, acc, cw=1.30):
    m = np.maximum.reduce([np.clip(tech, 0, 1), np.clip(eth, 0, 1),
                           np.clip(acc, 0, 1)])
    return np.minimum(1.0, cw * m)


def naive_avg_escalates(tech, eth, acc, cw=1.30):
    m = (np.clip(tech, 0, 1) + np.clip(eth, 0, 1) + np.clip(acc, 0, 1)) / 3.0
    return np.minimum(1.0, cw * m) >= HUMAN_REVIEW_BOUNDARY


def response_tier(governance_risk):
    """Vectorized band index 0..5 for a governance_risk array."""
    g = np.asarray(governance_risk)
    edges = np.array([0.20, 0.35, 0.50, 0.70, 0.85])
    return np.searchsorted(edges, g, side="right")


def wilson_ci(k, n, z=1.959963984540054):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
