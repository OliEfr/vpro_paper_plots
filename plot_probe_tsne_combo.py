"""Probe figure (left) beside the t-SNE embodiment-mixing 2x2 (right), one \\textwidth float.

    python plot_probe_tsne_combo.py                  # figures/probe_tsne_combo.pdf
    python plot_probe_tsne_combo.py --style science
    python plot_probe_tsne_combo.py --probe ridge

Three regions on one page:

    [ per-dim R^2 bars ][ R^2 vs SR scatter ] | [ 1k ][ 70k ]  2 emb
    ~~~~~~~~ plot_probe_combo, one probe ~~~~~ | [ 1k ][ 70k ]  5 emb

LEFT + MIDDLE reproduce plot_probe_combo.py for the chosen probe (default
MLP): bars are R^2 per action dimension averaged over benchmarks for every
learned teacher; the scatter is R^2 against the paper-table success rate with
colour = method, marker = benchmark. RIGHT is plot_tsne_teachers_2x2.py: the
2-embodiment and 5-embodiment teachers' latent t-SNE at 1k and 70k steps,
colour = robot.

EVERYTHING IS IMPORTED, NOT COPIED: bar geometry, method colours and hatches
from plot_probe_combo; the t-SNE panel routine, robot order, rows/cols and
per-row mark sizes from plot_tsne_teachers_2x2; loaders from plot_probe_perdim
and plot_probe_sr. Changing a colour or a mark size there changes it here.

WIDTH BUDGET. The t-SNE block is authored square-ish at about a third of the
width; the probe pair takes the remaining two thirds. Both are drawn at 8pt and
the page is exactly \\textwidth, so nothing is rescaled on include. Two legend
strips: methods + benchmarks above the probe pair, robots above the t-SNE.
"""

from __future__ import annotations

import argparse
import importlib

import pandas as pd

import plot_probe_combo as combo
import plot_probe_perdim as perdim
import plot_probe_sr as srbase
import plot_probe_sr_methods as srm
import plot_tsne_teachers_2x2 as tsne

# Share of the width given to the t-SNE block. Derived in make_figure so that a
# square panel exactly fills its grid cell: the block is then as tall as the
# probe axes and no wider than its two squares.
TSNE_FRAC = None
ROBOT_LABELS = {"franka": "Franka", "iiwa": "IIWA", "kinova3": "Kinova3", "ur5e": "UR5e",
                "sawyer": "Sawyer"}
# Robot colours for THIS figure: iiwa and kinova3 exchange their palette slots
# so franka (blue) and iiwa are no longer the palette's closest pair on the
# 2-embodiment row. The standalone t-SNE figures keep the family mapping.
ROBOT_SLOTS = {"franka": 0, "iiwa": 2, "kinova3": 1, "ur5e": 3}


def make_figure(probe, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pkey, rcol, plabel = combo.PROBES[probe]
    bars = perdim.averaged(perdim.load("current_action"))
    pts = srbase.load("sr_paper", all_arms=False)
    tdf = pd.read_csv(tsne.DEFAULT_CSV, dtype={"checkpoint": str})
    pal = list(style.PALETTE) + ["#000000"]
    tsne.EMB_COLORS.update({n: pal[ROBOT_SLOTS.get(n, 4)] for n in tsne.EMB_ORDER})
    tsne.style = style

    w = style.TEXT_WIDTH
    h = w * 0.315   # 0.42 * 0.75
    L, R, B, T, OUTER_WS, TS_WS, TS_HS = 0.085, 0.995, 0.15, 0.77, 0.16, 0.08, 0.10
    # t-SNE share of the width such that each square panel exactly fills its
    # cell (so the block's height equals the probe axes' height): panel side =
    # region height / (2 + hspace); block = 2 squares + wspace; the outer
    # wspace is a fraction of the mean column width, hence the 1 + OUTER_WS/2.
    region_h_in = (T - B) * h
    side_in = region_h_in / (2 + TS_HS)
    block_in = side_in * (2 + TS_WS)
    region_w_in = (R - L) * w
    frac = block_in * (1 + OUTER_WS / 2) / region_w_in
    global TSNE_FRAC
    TSNE_FRAC = frac
    fig = plt.figure(figsize=(w, h))
    # outer grid: probe pair | t-SNE block
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1 - frac, frac],
                     left=L, right=R, bottom=B, top=T, wspace=OUTER_WS)
    left = outer[0].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.24)
    right = outer[1].subgridspec(2, 2, wspace=TS_WS, hspace=TS_HS)

    axb = fig.add_subplot(left[0])
    axs = fig.add_subplot(left[1])
    combo.draw_bars(axb, bars, pkey, style)
    axb.set_ylabel(rf"{plabel.split()[0]} $R^2$" + "\n(mean over benchmarks)")
    axb.set_xlabel("Action Dimension")
    srm.OURS = combo.OURS
    srm.draw(axs, pts, rcol, style, "absolute", "none", note_loc="none",
             colors=combo.COLORS, edge_ours=style.MARKER_EDGE)
    axs.set_xlabel(rf"{plabel.split()[0]} $R^2$")
    axs.set_ylabel("Policy SR [%]")

    taxes = [[fig.add_subplot(right[i, j]) for j in range(2)] for i in range(2)]
    for i, (key, row_label, size) in enumerate(tsne.ROWS):
        sub_all = tdf[tdf["teacher"] == key]
        for j, (ckpt, _) in enumerate(tsne.COLS):
            tsne.panel(taxes[i][j], sub_all[sub_all["checkpoint"] == ckpt], size)
            taxes[i][j].set_box_aspect(1)   # square panels, whatever the data range
        # short row labels: the panels are too low here for "N embodiments"
        taxes[i][0].set_ylabel(row_label.replace("embodiments", "emb."), labelpad=2)
    for j, (_, col_label) in enumerate(tsne.COLS):
        # below the bottom row, at body text size (a title would pick up
        # axes.titlesize, which the science style sets larger)
        taxes[1][j].set_xlabel(col_label, labelpad=3)

    # legends: methods (two rows) + benchmarks over the probe pair; robots over the t-SNE
    method_handles = [Patch(facecolor=combo.COLORS[k], hatch=combo.HATCHES[k],
                            edgecolor=style.MARKER_EDGE, linewidth=0.35, label=lab)
                      for k, lab in srm.METHODS]
    bench_handles = [Line2D([], [], linestyle="none", marker=style.MARKERS[i], markersize=5,
                            markerfacecolor="white", markeredgecolor=style.INK,
                            markeredgewidth=0.6, label=lab)
                     for i, (_, lab) in enumerate(srbase.SUITES)]
    robot_handles = [Line2D([0], [0], marker="o", color="none",
                            markerfacecolor=tsne.EMB_COLORS[n], markeredgecolor=style.MARKER_EDGE,
                            markeredgewidth=0.4, markersize=3.4, label=ROBOT_LABELS.get(n, n))
                     for n in tsne.EMB_ORDER]
    # methods: flush left over the probe pair (2 rows x 4); benchmarks: inside
    # the scatter, top-left; robots: flush right over the t-SNE block (2 rows x 3)
    fig.legend(handles=method_handles, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 1.005),
               frameon=False, handletextpad=0.35, columnspacing=0.8, handlelength=1.2)
    axs.legend(handles=bench_handles, loc="upper left", frameon=False, handletextpad=0.3,
               labelspacing=0.15, borderaxespad=0.2)
    fig.legend(handles=robot_handles, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.005),
               frameon=False, handletextpad=0.15, columnspacing=0.5, handlelength=0.9)
    # vertical rule separating the probe pair from the embodiment-mixing block
    from matplotlib.lines import Line2D as _L
    fig.canvas.draw()   # box_aspect is applied at draw time; positions are final after this
    # midway between the scatter and the t-SNE row labels (~0.035 of the width)
    xsep = (axs.get_position().x1 + taxes[0][0].get_position().x0 - 0.035) / 2
    fig.add_artist(_L([xsep, xsep], [0.0, 1.0], transform=fig.transFigure,
                      color=style.INK_MUTED, linewidth=0.6))
    return style.save(fig, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", choices=sorted(combo.PROBES), default="mlp")
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    style = importlib.import_module("style" if a.style == "paper" else "style_science")
    style.apply_style()
    name = "probe_tsne_combo" + ("" if a.probe == "mlp" else f"_{a.probe}")
    name += "_science" if a.style == "science" else ""
    make_figure(a.probe, style, name)


if __name__ == "__main__":
    main()
