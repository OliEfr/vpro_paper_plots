r"""Real-world latent action probing, per action dimension -- single- vs multi-view LAM.

Reads ``results/probe_perdim_realworld.csv`` (written by
``experiments/extract_probe_realworld.py`` from Felix's DK1 probe outputs on MN5)
and emits ``figures/probe_perdim_realworld[_science].pdf``.

Real-hardware twin of the left panel of the paper's probing figure: R^2 of a
probe that reconstructs the 7-D end-effector action from the frozen 8-D latent,
scored on held-out robot episodes (episode split), for the two LAMs behind the
real-world multi-view-vs-single-view rollout comparison:

    ours_single  side-only DINO768d6 LAM   (sharedlam3, job 44160743, 30k)
    ours_multi   front+side DINO768d6 LAM  (sharedlam2, job 43524549, 30k)

Two targets exist in the CSV. ``current_action`` is the cluster probe on the stored DK1
action, which is an ABSOLUTE end-effector pose target (identical 119,983 / 36,067
train / test rows for both LAMs); it scores pose decodability. ``state_delta_h5``
(default) is our own probe on the realized EE motion state[t+5]-state[t] over the
LAM horizon, the real-world analogue of the simulation suites' delta actions
(20 %-of-episodes holdout, ridge alpha=1 / MLP(512,256), fit_decode_realworld.py).
Layout mirrors plot_probe_perdim.py: ridge block on the left, MLP block on the
right, a "Mean" group first in each block, then the seven dimensions.

Usage:
    python plot_probe_perdim_realworld.py [--style paper|science]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style
import plot_probe_combo as combo

CSV = Path(__file__).resolve().parent / "results" / "probe_perdim_realworld.csv"

PROBES = [("ridge", "Ridge probe (linear)"), ("mlp", "MLP probe (nonlinear)")]
METHODS = [("ours_single", "Ours (single-view, side)"), ("ours_multi", "Ours (multi-view, front + side)")]
# DK1 actions are ABSOLUTE end-effector pose targets (x, y, z, rotvec, gripper), not
# per-step deltas as in the simulation suites, so the dims are labelled as poses here.
# The stored DK1 action is an ABSOLUTE end-effector pose target, so the cluster probe
# ("current_action") scores pose decodability. The default target here is the local
# motion probe "state_delta_h5" (EE displacement over the LAM horizon of 5 frames),
# the real-world analogue of the per-step delta actions of the simulation suites.
GROUPS = {"state_delta_h5": [("mean", "Mean"), ("dx", r"$\Delta x$"), ("dy", r"$\Delta y$"), ("dz", r"$\Delta z$"),
                             ("droll", r"$\Delta\theta_x$"), ("dpitch", r"$\Delta\theta_y$"),
                             ("dyaw", r"$\Delta\theta_z$"), ("gripper", r"$\Delta$grip")],
          "current_action": [("mean", "Mean"), ("dx", "$x$"), ("dy", "$y$"), ("dz", "$z$"),
                             ("droll", r"$\theta_x$"), ("dpitch", r"$\theta_y$"), ("dyaw", r"$\theta_z$"),
                             ("gripper", "grip")]}
XLABEL = {"state_delta_h5": "Action Dimension (EE motion over 5 frames)",
          "current_action": "Action Dimension (absolute EE pose target)"}
BAR_W, GAP_MEAN, GAP_BLOCK = 0.36, 0.35, 1.2


def load(target):
    df = pd.read_csv(CSV)
    df = df[(df.target == target) & df.method.isin([m for m, _ in METHODS])]
    return df.set_index(["probe", "method", "dim_name"]).r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--target", choices=["state_delta_h5", "current_action"], default="state_delta_h5")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt

    r2 = load(a.target)
    groups = GROUPS[a.target]
    fig, ax = plt.subplots(figsize=style.figsize("text", ratio=0.30))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.76, bottom=0.2)
    x0 = 0.0
    ticks, labels, seps = [], [], []
    for pi, (pkey, plabel) in enumerate(PROBES):
        block_start = x0
        for gi, (gkey, glabel) in enumerate(groups):
            for mi, (mkey, _) in enumerate(METHODS):
                v = r2[(pkey, mkey, gkey)]
                xb = x0 + (mi - (len(METHODS) - 1) / 2) * BAR_W
                ax.bar(xb, v, BAR_W, facecolor=combo.COLORS[mkey], edgecolor=style.MARKER_EDGE,
                       linewidth=0.5, zorder=3)
                ax.text(xb, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=style.FIG_PT * 0.75 if hasattr(style, "FIG_PT") else 6, rotation=90)
            ticks.append(x0)
            labels.append(glabel)
            if gi == 0:
                seps.append(x0 + 0.5 + GAP_MEAN / 2)
                x0 += 1 + GAP_MEAN
            else:
                x0 += 1
        ax.text((block_start + x0 - 1) / 2, 1.06, plabel, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
        if pi == 0:
            blk = x0 - 0.5 + GAP_BLOCK / 2
            ax.axvline(blk, color=style.INK, linewidth=0.8, zorder=2)
            x0 += GAP_BLOCK
    for s in seps:
        ax.axvline(s, color=style.INK_MUTED, linewidth=0.6, linestyle=":", zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, x0 - 1 + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$R^2$ (real robot, held-out eps.)")
    ax.set_xlabel(XLABEL[a.target])
    ax.grid(axis="y", color=style.GRID, linewidth=0.5, zorder=0)
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=combo.COLORS[m], edgecolor=style.MARKER_EDGE, linewidth=0.5) for m, _ in METHODS],
               [l for _, l in METHODS], loc="upper left", ncol=2, frameon=False,
               bbox_to_anchor=(0.06, 1.0), handletextpad=0.5, columnspacing=1.5)
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"probe_perdim_realworld{'' if a.target == 'state_delta_h5' else '_abs'}{suffix}")


if __name__ == "__main__":
    main()
