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
colour = method, marker = benchmark. The two villa-X arms (scatter-only in
plot_probe_combo) are dropped here so one method legend serves both panels.
RIGHT is plot_tsne_teachers_2x2.py: the 2-embodiment and 5-embodiment
teachers' latent t-SNE at 1k and 70k steps, colour = robot.

EVERYTHING IS IMPORTED, NOT COPIED: bar geometry, method colours and hatches
from plot_probe_combo; the t-SNE panel routine, robot order, rows/cols and
per-row mark sizes from plot_tsne_teachers_2x2; loaders from plot_probe_perdim
and plot_probe_sr. Changing a colour or a mark size there changes it here.

HEIGHT BUDGET. The page is HEIGHT_FRAC of \\textwidth tall. The t-SNE block
runs the FULL page height (no legend strip above it) with square panels, and
its width follows from that; the robot key is rotated text to its right
(ROBOT_KEY_W), and the probe pair takes whatever width remains, capped below
the method legend (T). Everything is drawn at 8pt and the page is exactly
\\textwidth, so nothing is rescaled on include.
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
# Scatter-only baselines dropped from this figure (they have no per-dim probe,
# so they never appear in the bars): one legend then serves both probe panels.
DROP_METHODS = {"villax_cont", "villax_vq"}
# Shorter legend labels for this figure only (the encoder detail is in the text).
LABEL_OVERRIDES = {"dino": "UniVLA-style", "lapa": "LAPA-pretrained"}
HEIGHT_FRAC = 0.24        # page height / textwidth
ROBOT_KEY_W = 0.30        # inches: two lines of rotated robot names right of the t-SNE
# All three keys are set this much below the axis text so the six-entry method
# key fits one row left of the separating rule.
LEGEND_SCALE = 0.9
SEP_IN = 0.34             # inches between the scatter and the t-SNE block (rule + row labels)
# robot names split across the two rotated lines, bottom-to-top within a line
ROBOT_KEY_LINES = [["franka", "iiwa", "kinova3"], ["ur5e", "sawyer"]]


def robot_key(fig, style, x0, y0, y1, width):
    """Robot colour key as rotated text: `len(ROBOT_KEY_LINES)` vertical lines
    of "o Name" pairs reading bottom-to-top, centred on [y0, y1] and laid out
    across [x0, x0 + width] in figure fractions. Measured with the renderer so
    the items pack without overlap regardless of font."""
    import plot_tsne_teachers_2x2 as tsne
    from matplotlib.lines import Line2D
    renderer = fig.canvas.get_renderer()
    fh_px = fig.get_size_inches()[1] * fig.dpi
    gap = 0.045                   # fig fraction between items within a line
    mk_pad = 0.028                # marker centre -> text start
    n = len(ROBOT_KEY_LINES)
    xs = [x0 + width * (k + 0.5) / n for k in range(n)]
    lines = []
    for x, names in zip(xs, ROBOT_KEY_LINES):
        texts, lens = [], []
        for name in names:
            t = fig.text(x, 0, ROBOT_LABELS.get(name, name), rotation=90, ha="center",
                         va="bottom", fontsize=legend_pt())
            bb = t.get_window_extent(renderer)
            texts.append(t)
            lens.append(bb.height / fh_px)
        lines.append((x, names, texts, lens,
                      sum(lens) + len(names) * mk_pad + (len(names) - 1) * gap))
    # all lines start at the same baseline: the longest one centred on [y0, y1]
    y_start = (y0 + y1) / 2 - max(t for *_, t in lines) / 2
    for x, names, texts, lens, _ in lines:
        y = y_start
        for name, t, ln in zip(names, texts, lens):
            fig.add_artist(Line2D([x], [y], marker="o", color="none", transform=fig.transFigure,
                                  markerfacecolor=tsne.EMB_COLORS[name],
                                  markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4,
                                  markersize=3.4))
            t.set_position((x, y + mk_pad))
            y += mk_pad + ln + gap


def plt_rc(key):
    import matplotlib
    from matplotlib import font_manager
    v = matplotlib.rcParams[key]
    return font_manager.FontProperties(size=v).get_size_in_points()


def legend_pt():
    return plt_rc("legend.fontsize") * LEGEND_SCALE


def make_figure(probe, style, name):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator

    pkey, rcol, plabel = combo.PROBES[probe]
    bars = perdim.averaged(perdim.load("current_action"))
    pts = srbase.load("sr_paper", all_arms=False)
    tdf = pd.read_csv(tsne.DEFAULT_CSV, dtype={"checkpoint": str})
    pal = list(style.PALETTE) + ["#000000"]
    tsne.EMB_COLORS.update({n: pal[ROBOT_SLOTS.get(n, 4)] for n in tsne.EMB_ORDER})
    tsne.style = style

    w = style.TEXT_WIDTH
    h = w * HEIGHT_FRAC
    L, R, B, T = 0.085, 0.995, 0.185, 0.835      # T caps the PROBE axes only (one-row method legend above)
    RT = 0.985                                   # the t-SNE block runs to the top of the page
    TS_WS, TS_HS = 0.08, 0.10
    # t-SNE geometry in inches: square panels fill the full page height, the
    # robot key hangs to their right, and the probe pair gets whatever is left.
    side_in = (RT - B) * h / (2 + TS_HS)
    block_in = side_in * (2 + TS_WS)
    key_in = ROBOT_KEY_W                         # rotated robot names, two lines
    ts_x1 = R - key_in / w                       # block right edge (fig fraction)
    ts_x0 = ts_x1 - block_in / w
    probe_x1 = ts_x0 - SEP_IN / w                # room for the rule + "N emb." row labels
    global TSNE_FRAC
    TSNE_FRAC = (R - ts_x0) / (R - L)
    fig = plt.figure(figsize=(w, h))
    left = GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=0.20,
                    left=L, right=probe_x1, bottom=B, top=T)
    right = GridSpec(2, 2, figure=fig, wspace=TS_WS, hspace=TS_HS,
                     left=ts_x0, right=ts_x1, bottom=B, top=RT)

    axb = fig.add_subplot(left[0])
    axs = fig.add_subplot(left[1])
    combo.draw_bars(axb, bars, pkey, style)
    # second line kept shorter than the axes are tall (the page is low)
    axb.set_ylabel(rf"{plabel.split()[0]} $R^2$" + "\n(benchmark mean)")
    axb.set_xlabel("Action Dimension", labelpad=1.5)
    srm.OURS = combo.OURS
    # villa-X is a scatter-only baseline; without it the panel is the six bar
    # methods and one legend serves both.
    pts = pts[~pts.method.isin(DROP_METHODS)]
    methods = [(k, LABEL_OVERRIDES.get(k, lab)) for k, lab in srm.METHODS if k not in DROP_METHODS]
    srm.draw(axs, pts, rcol, style, "absolute", "none", note_loc="none",
             colors=combo.COLORS, edge_ours=style.MARKER_EDGE)
    axs.set_xlabel(rf"{plabel.split()[0]} $R^2$", labelpad=1.5)
    axs.xaxis.set_major_locator(MultipleLocator(0.1))
    axs.set_ylabel("Policy SR [%]")

    taxes = [[fig.add_subplot(right[i, j]) for j in range(2)] for i in range(2)]
    for i, (key, row_label, size) in enumerate(tsne.ROWS):
        sub_all = tdf[tdf["teacher"] == key]
        for j, (ckpt, _) in enumerate(tsne.COLS):
            tsne.panel(taxes[i][j], sub_all[sub_all["checkpoint"] == ckpt], size)
            taxes[i][j].set_box_aspect(1)   # square panels, whatever the data range
        taxes[i][0].set_ylabel(row_label.replace("embodiments", "emb."), labelpad=2)
    for j, (_, col_label) in enumerate(tsne.COLS):
        # below the bottom row, at body text size (a title would pick up
        # axes.titlesize, which the science style sets larger)
        taxes[1][j].set_xlabel(col_label, labelpad=2)

    # legends: methods in one row over the probe pair; benchmarks inside the
    # scatter; robots as rotated text beside the t-SNE block (see robot_key)
    method_handles = [Patch(facecolor=combo.COLORS[k], hatch=combo.HATCHES[k],
                            edgecolor=style.MARKER_EDGE, linewidth=0.35, label=lab)
                      for k, lab in methods]
    bench_handles = [Line2D([], [], linestyle="none", marker=style.MARKERS[i], markersize=5,
                            markerfacecolor="white", markeredgecolor=style.INK,
                            markeredgewidth=0.6, label=lab)
                     for i, (_, lab) in enumerate(srbase.SUITES)]
    # lower right: with villa-X gone the points sit in the upper-left half
    axs.legend(handles=bench_handles, loc="lower right", frameon=False, handletextpad=0.3,
               labelspacing=0.15, borderaxespad=0.2, fontsize=legend_pt())
    fig.canvas.draw()   # box_aspect is applied at draw time; positions are final after this
    robot_key(fig, style, x0=ts_x1 + 0.004, y0=B, y1=RT, width=key_in / w)
    # vertical rule separating the probe pair from the embodiment-mixing block:
    # midway between the scatter's right edge and the "N emb." row label
    lab_x0 = min(ax[0].yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0
                 for ax in taxes) / (w * fig.dpi)
    xsep = (axs.get_position().x1 + lab_x0) / 2
    fig.add_artist(Line2D([xsep, xsep], [0.0, 1.0], transform=fig.transFigure,
                          color=style.INK_MUTED, linewidth=0.6))
    # one row, centred on everything left of the rule: the key belongs to both
    # probe panels, and the probe pair alone is narrower than six entries
    fig.legend(handles=method_handles, loc="upper center", ncol=len(method_handles),
               bbox_to_anchor=(xsep / 2, 1.005), frameon=False, handletextpad=0.3,
               columnspacing=0.6, handlelength=1.0, fontsize=legend_pt())
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
