"""Generates Figs. 4-6 of the paper from results/raw/*.json.

No titles are drawn inside the figures: per IEEE style the title belongs
in the typeset caption, and repeating it in the artwork duplicates the
caption in a competing font.

Fig. 4: threshold sensitivity on the borderline/noisy scenario.
Fig. 5: response accuracy by scenario type.
Fig. 6: the borderline/noisy-scenario trade-off.

Writes PNGs to figures/.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RED = "#C44E52"
BLUE = "#4C72B0"
GREEN = "#55A868"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "results", "raw")
FIGS = os.path.join(HERE, "..", "figures")


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def fig4():
    with open(os.path.join(RAW, "borderline.json")) as f:
        sweep = json.load(f)["threshold_sweep"]
    taus = [r["tau"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    ax.plot(taus, [r["false_escalation"] for r in sweep], "-o", color=RED,
            label="False escalation rate")
    ax.plot(taus, [r["missed_critical"] for r in sweep], "-s", color=BLUE,
            label="Missed critical rate")
    ax.plot(taus, [r["balanced_accuracy"] for r in sweep], "--^",
            color=GREEN, label="Balanced accuracy")
    ax.axvline(0.20, color="gray", linestyle=":", linewidth=1.2)
    ax.text(0.205, 0.02, "default \u03c4=0.20", color="gray", fontsize=9)
    ax.set_xlabel("Dominance threshold \u03c4")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, loc="center left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_threshold_sensitivity.png"))
    plt.close(fig)


def fig5():
    with open(os.path.join(RAW, "scenarios.json")) as f:
        t6 = json.load(f)["table6"]
    names = ["technical_driven", "ethical_driven", "accountability_driven",
             "combined_moderate", "all_low", "all_high"]
    labels = [n.replace("_", "\n") for n in names]
    pavi = [t6[n]["pavi_acc"] for n in names]
    igra = [t6[n]["igra_acc"] for n in names]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)
    ax.bar([i - w / 2 for i in x], pavi, w, color=RED, label="PAVI-only")
    ax.bar([i + w / 2 for i in x], igra, w, color=BLUE, label="IGRA")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Response accuracy")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_scenario_accuracy.png"))
    plt.close(fig)


def fig6():
    with open(os.path.join(RAW, "borderline.json")) as f:
        t8 = json.load(f)["table8"]
    metrics = [("accuracy", "Accuracy"),
               ("false_escalation", "False escalation\nrate"),
               ("missed_critical", "Missed critical\nrate")]
    pavi = [t8["pavi"][m] for m, _ in metrics]
    igra = [t8["igra"][m] for m, _ in metrics]
    x = range(len(metrics))
    w = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    b1 = ax.bar([i - w / 2 for i in x], pavi, w, color=RED,
                label="PAVI-only")
    b2 = ax.bar([i + w / 2 for i in x], igra, w, color=BLUE, label="IGRA")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_borderline_tradeoff.png"))
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig4()
    fig5()
    fig6()
    print("Wrote figures to", os.path.abspath(FIGS))


if __name__ == "__main__":
    main()
