r"""The DK1 hardware trade-off: embodiment overlap against what the latent still knows.

Reads ``results/xemb_hardware_frontier.csv`` (written by
``experiments/collect_xemb_hardware_frontier.py``) and emits
``figures/xemb_hardware_frontier[_science].pdf``.

Every intervention tried on the real-hardware latent lands on one downward frontier: the more
robot demos and human video overlap, the less the latent knows about the action. This figure
is where a new arm lands on it.

Axes. x is source decodability on the raw latent (logistic regression, chance 0.5): how
easily robot and human-video frames can be told apart, so LEFT is more embodiment-invariant.
y is action R^2, an episode-split ridge from the latent to the 7-D end-effector action on
robot rows, so UP is a latent that still carries the action. The useful corner is TOP LEFT,
and it is empty for every learned arm measured so far.

Marker area encodes the latent's mean per-dimension standard deviation, because the cheapest
way to reach the left edge is to destroy the latent: the CORAL arms sit there with standard
deviations near 0.003 and 0.0006, which is collapse, not invariance. A tiny marker is a
warning, not a result.

The post-hoc point is drawn in a different shape on purpose. It is the baseline latent with
per-source standardisation applied at analysis time, so it uses the source label and is not a
learned invariant latent. It marks what the geometry permits: the two sources are the same
manifold offset in mean and scale, and removing that offset costs no action content.

Usage:
    python plot_xemb_hardware_frontier.py [--style paper|science]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "xemb_hardware_frontier.csv"

# Label placement, per arm. Three of the teachers sit almost on top of each other at the
# top right, so those get an explicit position in DATA coordinates plus a leader line; the
# rest are fine with a small offset in points.
#   arm -> (position, ha, leader)  where position is (x, y) in data coords if leader else
#   an (dx, dy) offset in points
PLACE = {
    "base_balanced":        ((0.815, 0.3190), "right", True),
    "activeruns":           ((0.815, 0.2805), "right", True),
    "base_5k":              ((0.815, 0.2420), "right", True),
    "coralstd3k":           ((-8, 0), "right", False),
    "dim4":                 ((0, -13), "center", False),
    "srcoffsets":           ((-8, -1), "right", False),
    "coral3k":              ((7, 2), "left", False),
    "coralstd_w003":        ((6, 10), "left", False),
    "base_balanced_srcstd": ((0, -14), "center", False),
    "learned":              ((0, 13), "center", False),
    "sharedlam6_learned":   ((7, 6), "left", False),
}


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
    from matplotlib.lines import Line2D

    df = pd.read_csv(CSV)
    fs = plt.rcParams["font.size"]
    fig, ax = plt.subplots(figsize=style.figsize("col", ratio=0.82))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.135)

    # marker area from latent std: collapse must be visible, not hidden in a footnote
    std = df.latent_std.to_numpy()
    size = 12 + 150 * np.sqrt(np.clip(std, 0, None) / max(std.max(), 1e-9))

    kinds = {"teacher": dict(color=style.PALETTE[0], marker="o", edge=style.MARKER_EDGE,
                             label="teacher, as trained"),
             "posthoc": dict(color=style.PALETTE[2], marker="s", edge=style.MARKER_EDGE,
                             label="post-hoc alignment (uses the label)"),
             "ours": dict(color=style.PALETTE[3] if len(style.PALETTE) > 3 else style.PALETTE[1],
                          marker="X", edge=style.MARKER_EDGE_OURS, label="ours (learned tokenizer)")}
    for kind, kw in kinds.items():
        m = (df.kind == kind).to_numpy()
        if not m.any():
            continue
        ax.scatter(df.source_logreg[m], df.action_r2[m], s=size[m], marker=kw["marker"],
                   facecolor=kw["color"], edgecolor=kw["edge"], linewidth=0.7, zorder=3)

    for _, r in df.iterrows():
        pos, ha, leader = PLACE.get(r.arm, ((6, 4), "left", False))
        kw = dict(ha=ha, va="center", fontsize=fs * 0.85, color=style.INK_MUTED,
                  annotation_clip=False)
        if leader:
            ax.annotate(r.label, (r.source_logreg, r.action_r2), xytext=pos, textcoords="data",
                        arrowprops=dict(arrowstyle="-", color=style.INK_MUTED, linewidth=0.4,
                                        shrinkA=1, shrinkB=3), **kw)
        else:
            ax.annotate(r.label, (r.source_logreg, r.action_r2), xytext=pos,
                        textcoords="offset points", **kw)

    ax.set_xlabel("source decodability (logistic regression)")
    ax.set_ylabel("action $R^2$ from the latent")
    ax.set_xlim(0.38, 1.045)
    ax.set_ylim(-0.02, 0.345)
    ax.axvline(0.5, color=style.INK_MUTED, linewidth=0.6, linestyle="--", zorder=1)
    ax.annotate("chance", xy=(0.5, 0.345), xytext=(3, -2), textcoords="offset points",
                ha="left", va="top", fontsize=fs * 0.85, color=style.INK_MUTED)
    ax.grid(color=style.GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [Line2D([0], [0], linestyle="none", marker=kw["marker"], markerfacecolor=kw["color"],
                      markeredgecolor=kw["edge"], markeredgewidth=0.7, markersize=5, label=kw["label"])
               for kind, kw in kinds.items() if (df.kind == kind).any()]
    # the free region is mid-left, between the post-hoc point and the CORAL arms
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 0.66),
              fontsize=fs * 0.9, handletextpad=0.4, borderaxespad=0.0)
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_hardware_frontier{suffix}")


if __name__ == "__main__":
    main()
