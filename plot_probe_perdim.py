"""Per-action-dimension latent->action probe R^2, four teachers, three benchmarks.

    python plot_probe_perdim.py                    # figures/probe_perdim.pdf
    python plot_probe_perdim.py --style science    # figures/probe_perdim_science.pdf
    python plot_probe_perdim.py --target future    # future_action_mean instead

Reads results/probe_perdim.csv (written by experiments/extract_probe_perdim.py)
and nothing else. Independent of plot_probing.py -- different dump, different
arms table, different layout; do not merge them.

LAYOUT. One axis, a two-column (\\textwidth) float, two blocks side by side:
the ridge (linear) probe on the left, the MLP (nonlinear) probe on the right.
Inside each block the x axis is the mean over the 7 action dimensions (set
apart by a dashed divider, because it is the headline number every table
quotes) followed by the 7 dimensions themselves (robosuite OSC_POSE deltas:
xyz, axis-angle rotation theta_xyz, gripper). Each group holds one plain bar
per teacher.

Every bar is the UNWEIGHTED MEAN OVER THE THREE BENCHMARKS (LIBERO,
LIBERO-Plus, MimicGen) of that teacher's R^2 on that dimension -- one number
per (probe, dimension, teacher). The per-benchmark values stay in
results/probe_perdim.csv; the printed table lists them next to the average.
The y axis is R^2 on [0, 1] for both blocks so a bar height means the same
thing in either probe; the rotation dimensions really are that low.

SERIES ORDER. Ours single-view, ours multi-view, CLAM, LAOF -- fixed, colour
and hatch assigned by position from style.PALETTE / style.HATCHES. The two
"ours" bars additionally carry the red edge that style.py reserves for the
hero configuration, so they can be told from the baselines in grayscale even
before reading the hatches.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parent / "results" / "probe_perdim.csv"

# The benchmarks averaged over (all three must be present for every cell).
SUITES = [("libero", "LIBERO"), ("libero_plus", "LIBERO-Plus"), ("mimicgen", "MimicGen")]
# Block order along the x axis: linear probe left, nonlinear right.
PROBES = [("ridge", "Ridge probe (linear)"), ("mlp", "MLP probe (nonlinear)")]
# Series order -- positional in PALETTE / HATCHES.
METHODS = [("ours_single", "Ours (single-view)"), ("ours_multi", "Ours (multi-view)"),
           ("clam", "CLAM"), ("laof", "LAOF")]
OURS = {"ours_single", "ours_multi"}
# x groups: the mean first, then the seven dims in action order.
GROUPS = [("mean", "Mean"), ("dx", r"$\Delta x$"), ("dy", r"$\Delta y$"), ("dz", r"$\Delta z$"),
          ("droll", r"$\Delta\theta_x$"), ("dpitch", r"$\Delta\theta_y$"),
          ("dyaw", r"$\Delta\theta_z$"), ("gripper", "grip")]
TARGETS = {"current": "current_action", "future": "future_action_mean"}

BAR_W = 0.20          # 4 bars * 0.20 = 0.80 of the unit group pitch
GAP_MEAN = 0.35       # extra space between a block's Mean group and its dims
GAP_BLOCK = 1.2       # extra space between the ridge block and the MLP block


def load(target: str) -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[(df.role == "canonical") & (df.target == target)]
    # exactly one arm per (benchmark, method), and every benchmark present for
    # every (method, probe, dim) -- otherwise the cross-benchmark mean is not a
    # mean over the same three suites for every bar.
    arms = df.groupby(["benchmark", "method"]).arm_key.nunique()
    bad = arms[arms != 1]
    if len(bad):
        raise SystemExit(f"expected one canonical arm per cell, got:\n{bad}")
    cnt = df.groupby(["method", "probe", "dim_name"]).benchmark.nunique()
    if (cnt != len(SUITES)).any():
        raise SystemExit(f"benchmarks missing for some cells:\n{cnt[cnt != len(SUITES)]}")
    return df


def averaged(df: pd.DataFrame) -> pd.Series:
    """R^2 averaged over benchmarks, indexed by (probe, method, dim_name)."""
    return df.groupby(["probe", "method", "dim_name"]).r2.mean()


def group_x() -> np.ndarray:
    """Centre of every x group: the ridge block, then the MLP block."""
    xs = []
    x0 = 0.0
    for _ in PROBES:
        block = np.arange(len(GROUPS), dtype=float)
        block[1:] += GAP_MEAN
        xs.append(x0 + block)
        x0 = xs[-1][-1] + 1.0 + GAP_BLOCK
    return np.stack(xs)


def make_figure(df: pd.DataFrame, style, name: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    avg = averaged(df)
    w = style.TEXT_WIDTH
    h = w * 0.30
    fig, ax = plt.subplots(figsize=(w, h))

    x = group_x()
    n = len(METHODS)
    for pi, (pkey, plabel) in enumerate(PROBES):
        for mi, (mkey, _) in enumerate(METHODS):
            ys = np.array([avg.get((pkey, mkey, g), np.nan) for g, _ in GROUPS])
            ours = mkey in OURS
            ax.bar(x[pi] + (mi - (n - 1) / 2) * BAR_W, ys, width=BAR_W,
                   color=style.PALETTE[mi], hatch=style.HATCHES[mi],
                   edgecolor=style.MARKER_EDGE_OURS if ours else style.MARKER_EDGE,
                   linewidth=0.5 if ours else 0.35, zorder=3)
        # mean | dims divider inside the block, block title above it
        ax.axvline((x[pi, 0] + x[pi, 1]) / 2, color=style.INK_MUTED, linewidth=0.6,
                   linestyle=(0, (3, 2)), zorder=2)
        ax.text(x[pi].mean(), 1.02, plabel, ha="center", va="bottom",
                transform=ax.get_xaxis_transform())
        if pi > 0:
            ax.axvline((x[pi - 1, -1] + x[pi, 0]) / 2, color=style.INK_MUTED,
                       linewidth=0.6, zorder=2)

    ax.set_xlim(x[0, 0] - 0.6, x[-1, -1] + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r"$R^2$ (mean over benchmarks)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x.ravel())
    ax.set_xticklabels([lab for _, lab in GROUPS] * len(PROBES))
    ax.tick_params(axis="x", length=0)

    handles = [Patch(facecolor=style.PALETTE[i], hatch=style.HATCHES[i],
                     edgecolor=style.MARKER_EDGE_OURS if k in OURS else style.MARKER_EDGE,
                     linewidth=0.5 if k in OURS else 0.35, label=lab)
               for i, (k, lab) in enumerate(METHODS)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    # Margins reserved by hand: savefig.bbox is None (see style.py), so the
    # saved page is exactly figsize; the legend and block titles need their
    # own strip on top.
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.80)
    return style.save(fig, name)


def print_table(df: pd.DataFrame) -> None:
    """Headline means per benchmark and their cross-benchmark average."""
    m = df[df.dim_name == "mean"].pivot_table(index=["probe", "benchmark"],
                                              columns="method", values="r2")
    m = m[[k for k, _ in METHODS]]
    avg = m.groupby(level="probe").mean()
    avg.index = pd.MultiIndex.from_product([avg.index, ["AVERAGE"]], names=m.index.names)
    print(pd.concat([m, avg]).sort_index().round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--target", choices=sorted(TARGETS), default="current",
                   help="probe target: current_action (default) or future_action_mean")
    p.add_argument("--name", default="probe_perdim", help="output stem under figures/")
    a = p.parse_args()

    style = importlib.import_module("style" if a.style == "paper" else "style_science")
    style.apply_style()
    df = load(TARGETS[a.target])
    print_table(df)
    name = a.name + ("" if a.target == "current" else f"_{a.target}")
    if a.style == "science":
        name += "_science"
    make_figure(df, style, name)


if __name__ == "__main__":
    main()
