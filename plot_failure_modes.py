r"""Where rollouts fail, simulation and real world side by side -- stacked bars.

Reads ``results/failure_modes.csv`` (typed from the paper's Table III and Table IV)
and emits ``figures/failure_modes[_science].pdf``.

One stacked bar per arm; the bar is 100 % of rollouts, the bottom segment is the
success share and the segments above it are the failure stages. LIBERO reports
failures as a share of *failures* (Table IV); the CSV converts them to a share of
*rollouts* so the two domains sit on one axis.

The two domains label failures differently, so they are mapped onto one coarse
taxonomy that both support (``CATEGORY`` below):

    reach / plan   LIBERO "made no contact"       real "wrong plan"
    grasp          LIBERO "grasping"              real "missed grip"
    lift / drop    LIBERO "lift"                  real "dropped after grip"
    transport / place  LIBERO "transport" + "place"  real "target location error"

The message the figure carries: adding video shrinks the reach/plan failures in
both domains and leaves grasping as the dominant residual failure; only more
action data (LIBERO "full action") reduces grasping, and it pushes the remaining
failures into placement.

Usage:
    python plot_failure_modes.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

CSV = Path(__file__).resolve().parent / "results" / "failure_modes.csv"

CATEGORY = {
    "success": "success",
    "made no contact": "reach / plan", "wrong plan": "reach / plan",
    "grasping": "grasp", "missed grip": "grasp",
    "lift": "lift / drop", "dropped after grip": "lift / drop",
    "transport": "transport / place", "place": "transport / place",
    "target location error": "transport / place",
}
ORDER = ["success", "reach / plan", "grasp", "lift / drop", "transport / place"]
GROUPS = [("libero", "LIBERO (sim)", [("w/ video (ours)", "video\n(ours)"),
                                      ("oracle latent", "oracle\nlatent"),
                                      ("full action", "full\naction")]),
          ("real_novel_obj", "Real: novel obj.", [("action-only", "action\nonly"),
                                                  ("w/ video (ours)", "video\n(ours)")]),
          ("real_novel_place", "Real: novel place", [("action-only", "action\nonly"),
                                                     ("w/ video (ours)", "video\n(ours)")])]
BAR_W, STEP, GAP = 0.72, 1.12, 0.8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt

    df = pd.read_csv(CSV, comment="#")
    df["category"] = df.outcome.map(CATEGORY)
    agg = df.groupby(["domain", "arm", "category"]).pct_of_rollouts.sum()

    colors = {"success": "#BDBDBD", "reach / plan": style.PALETTE[0], "grasp": style.PALETTE[3],
              "lift / drop": style.PALETTE[2], "transport / place": style.PALETTE[1]}

    fig, ax = plt.subplots(figsize=style.figsize("col", ratio=0.8))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.74, bottom=0.18)
    x = 0.0
    ticks, labels = [], []
    for gi, (dom, dlabel, arms) in enumerate(GROUPS):
        gx0 = x
        for arm, alabel in arms:
            bottom = 0.0
            for cat in ORDER:
                v = float(agg.get((dom, arm, cat), 0.0))
                ax.bar(x, v, BAR_W, bottom=bottom, facecolor=colors[cat],
                       edgecolor=style.MARKER_EDGE, linewidth=0.4, zorder=3)
                if cat == "success":
                    ax.text(x, v / 2, f"{v:.0f}", ha="center", va="center", color=style.INK)
                bottom += v
            ticks.append(x)
            labels.append(alabel)
            x += STEP
        ax.text((gx0 + x - STEP) / 2, 1.02, dlabel, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
        if gi < len(GROUPS) - 1:
            ax.axvline(x - STEP / 2 + GAP / 2, color=style.INK_MUTED, linewidth=0.6, zorder=2)
            x += GAP
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.65, x - STEP + 0.65)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rollouts [%]")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[c], edgecolor=style.MARKER_EDGE, linewidth=0.4) for c in ORDER]
    fig.legend(handles, ORDER, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.56, 1.0), handletextpad=0.4, columnspacing=1.0)
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"failure_modes{suffix}")


if __name__ == "__main__":
    main()
