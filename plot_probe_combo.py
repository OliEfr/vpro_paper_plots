"""One probe, two views: per-dimension R^2 (left) and R^2 vs success rate (right).

    python plot_probe_combo.py --probe mlp     # figures/probe_combo_mlp.pdf
    python plot_probe_combo.py --probe ridge   # figures/probe_combo_ridge.pdf
    python plot_probe_combo.py --probe mlp --style science

Two-column float, two panels. LEFT is plot_probe_perdim.py's bar block for the
chosen probe only: R^2 per action dimension (mean over the three benchmarks,
Mean group set apart by the divider) for every learned teacher in the paper
table -- ours single/multi-view, CLAM, LAOF, DINOv3, LAPA -- i.e. everything
except the frozen villa-X latents. RIGHT is plot_probe_sr_methods.py's scatter
for the same probe: R^2 vs the success rate printed in the paper's main table,
one point per (method, benchmark), all eight table rows including villa-X.

COLOURS ARE THIS FIGURE'S OWN (COLORS below), not style.PALETTE: the two
"ours" teachers are the saturated reds and drawn solid, every baseline is a
muted tone with a hatch, villa-X is grey. So the eye lands on the reds and the
hatches still separate the baselines in grayscale. Marker shape = benchmark in
the scatter.

Everything else is IMPORTED, NOT COPIED: group order, bar geometry and the
averaging come from plot_probe_perdim; scatter encoding and the data loader
from plot_probe_sr_methods / plot_probe_sr. Change them there.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np

import plot_probe_perdim as perdim
import plot_probe_sr as srbase
import plot_probe_sr_methods as srm

PROBES = {"ridge": ("ridge", "ridge_r2", "Ridge probe (linear)"),
          "mlp": ("mlp", "mlp_r2", "MLP probe (nonlinear)")}

# Bar methods, in bar order; the scatter adds the two villa-X rows after them.
BAR_METHODS = [("ours_single", "Ours (single-view)"), ("ours_multi", "Ours (multi-view)"),
               ("clam", "CLAM-style"), ("laof", "LAOF-style"),
               ("dino", "UniVLA-style (DINOv3)"), ("lapa", "Ours (LAPA-pretrained)")]
COLORS = {
    "ours_single": "#E8833A",   # warm orange-red: ours, single-view
    "ours_multi": "#C8102E",    # strong red: ours, multi-view (the hero)
    "clam": "#8FA9C4",          # muted, low-saturation baselines
    "laof": "#9DBF9E",
    "dino": "#C2B280",
    "lapa": "#A99BC4",
    "villax_cont": "#7F7F7F",
    "villax_vq": "#BDBDBD",
}
HATCHES = {"ours_single": "", "ours_multi": "", "clam": "///", "laof": "...",
           "dino": "xxx", "lapa": "\\\\\\", "villax_cont": "", "villax_vq": ""}
OURS = {"ours_single", "ours_multi"}   # LAPA is a baseline here despite the "Ours" label
BAR_W = 0.13                            # 6 bars * 0.13 = 0.78 of the group pitch


def draw_bars(ax, avg, pkey, style):
    """One block of plot_probe_perdim's figure: Mean + 7 dims, six teachers."""
    x = np.arange(len(perdim.GROUPS), dtype=float)
    x[1:] += perdim.GAP_MEAN
    n = len(BAR_METHODS)
    for mi, (mkey, _) in enumerate(BAR_METHODS):
        ys = np.array([avg.get((pkey, mkey, g), np.nan) for g, _ in perdim.GROUPS])
        if np.isnan(ys).any():
            raise SystemExit(f"{mkey} missing on some benchmark for probe {pkey}")
        ax.bar(x + (mi - (n - 1) / 2) * BAR_W, ys, width=BAR_W,
               color=COLORS[mkey], hatch=HATCHES[mkey], edgecolor=style.MARKER_EDGE,
               linewidth=0.35, zorder=3)
    ax.axvline((x[0] + x[1]) / 2, color=style.INK_MUTED, linewidth=0.6,
               linestyle=(0, (3, 2)), zorder=2)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in perdim.GROUPS])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def make_figure(probe, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pkey, rcol, plabel = PROBES[probe]
    bars = perdim.averaged(perdim.load("current_action"))
    pts = srbase.load("sr_paper", all_arms=False)

    w = style.TEXT_WIDTH
    h = w * 0.38
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [1.15, 1]})

    draw_bars(axl, bars, pkey, style)
    axl.set_ylabel(rf"{plabel.split()[0]} $R^2$, mean over benchmarks")
    axl.set_title("per action dimension", pad=3)

    srm.OURS = OURS   # scatter: red edge only on the two hero teachers
    srm.draw(axr, pts, rcol, style, "absolute", "none", note_loc="upper left", colors=COLORS,
             edge_ours=style.MARKER_EDGE)
    axr.set_xlabel(rf"{plabel.split()[0]} $R^2$")
    axr.set_ylabel("success rate (%)")
    axr.set_title("vs. downstream success rate", pad=3)

    from matplotlib.patches import Patch
    method_handles = [Patch(facecolor=COLORS[k], hatch=HATCHES[k], edgecolor=style.MARKER_EDGE,
                            linewidth=0.35, label=lab) for k, lab in srm.METHODS]
    bench_handles = [
        Line2D([], [], linestyle="none", marker=style.MARKERS[i], markersize=5,
               markerfacecolor="white", markeredgecolor=style.INK, markeredgewidth=0.6, label=lab)
        for i, (_, lab) in enumerate(srbase.SUITES)]
    fig.legend(handles=method_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005),
               frameon=False, handletextpad=0.4)
    fig.legend(handles=bench_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.885),
               frameon=False, handletextpad=0.4)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.14, top=0.735, wspace=0.22)
    return style.save(fig, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", choices=sorted(PROBES), required=True)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    style = importlib.import_module("style" if a.style == "paper" else "style_science")
    style.apply_style()
    make_figure(a.probe, style, f"probe_combo_{a.probe}" + ("_science" if a.style == "science" else ""))


if __name__ == "__main__":
    main()
