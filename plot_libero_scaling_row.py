r"""The three LIBERO scaling sweeps as one figure*, a plotted twin of the paper's
scaling table (tab:data_scaling_combined).

Reads ``results/libero_scaling_row.csv`` and draws one row of three panels, one
per sweep -- (a) action-labeled Panda episodes per task, (b) % of LIBERO-90
used as play data, (c) number of cross-embodiments in the video pretraining
mix. Both task splits sit in the same axes: in-distribution as solid lines with
filled markers, held-out as dashed lines with hollow markers. Panel (c) adds the
single-view (sideview-only teacher) arm next to the dual-view one, from the
``sideview_sr`` column. Run with no arguments:

    python plot_libero_scaling_row.py

SCIENCE STYLE ONLY, like plot_realworld_row.py: it exists to be dropped into the
paper, which takes the SciencePlots variant, so build_figures.sh runs it once
(see SCIENCE_ONLY there).

EMBEDDING IN LATEX
------------------
Built at 7.140in = \textwidth, so a figure*, included at width=\textwidth:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/libero_scaling_row_science.pdf}
      \caption{}
      \label{fig:libero_scaling_row}
    \end{figure*}

The caption is the author's. What it has to carry, because the figure does not
say it: held-out tasks only ever see cross-embodiment videos, never action
data; panel (c) has no action-only line because action-only does not use the
video pretraining mix at all; panel (c)'s single-view arm is the sideview-only
teacher at the same recipe; and the cells at 5 episodes, 100% play data and
2 cross-embodiments (dual-view) are one and the same run.

TWO FACTORS, TWO CHANNELS (README.md). Method is colour plus marker shape, the
split is line style plus marker fill, and each factor gets its own legend row
so a reader can hold one fixed and scan the other. The three panels share one
0-85 y axis, labelled once on the left. The x positions are categorical and
evenly spaced -- the sweeps are not linear in their own units (1/5/10/20
episodes, 1/2/4 embodiments) and the table reads them as steps.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style_science as style

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV = RESULTS_DIR / "libero_scaling_row.csv"
OUT_NAME = "libero_scaling_row_science"

METHODS = ["action_only_sr", "video_sr", "sideview_sr"]
METHOD_LABELS = {"action_only_sr": "Action-only", "video_sr": "+ LAM dual-view (ours)",
                 "sideview_sr": "+ LAM single-view"}
# Split channel: in-distribution solid / filled, held-out dashed / hollow.
SPLIT_LINE = {"nonh": "-", "h": "--"}
SPLIT_LABELS = {"nonh": "In-distribution tasks", "h": "Held-out tasks"}

SWEEPS = ["budget", "play", "xemb"]
SWEEP_XLABELS = {
    "budget": "(a) # Episodes with action labels",
    "play": "(b) % of LIBERO-90 as play data",
    "xemb": "(c) # X-embodiments in video pretraining",
}
SPLITS = ["h", "nonh"]
YLIM = (0, 85)
YTICKS = [0, 20, 40, 60, 80]

# Bands in inches. Measured off the rendered text at 8pt, like the real-world
# row: two-line y labels plus tick labels on the left, one legend row on top,
# tick labels plus the one-line x label at the bottom.
YLABEL_IN = 0.40
GUTTER_IN = 0.22
RIGHT_PAD_IN = 0.08
LEGEND_ROW_IN = 0.13
LEGEND_ROWS = 2       # one row per factor: methods, then splits
LEGEND_GAP_IN = 0.05
XTICKS_IN = 0.34
BODY_IN = 1.05
FIG_HEIGHT_IN = LEGEND_ROWS * LEGEND_ROW_IN + LEGEND_GAP_IN + BODY_IN + XTICKS_IN


def load():
    df = pd.read_csv(CSV, comment="#")
    assert set(df["sweep"]) == set(SWEEPS), sorted(set(df["sweep"]))
    assert set(df["split"]) == set(SPLITS), sorted(set(df["split"]))
    return df


def style_for(i):
    return style.PALETTE[i % len(style.PALETTE)]


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = load()

    print("\n  LIBERO scaling sweeps, success rate %  (in-distribution / held-out)")
    for sw in SWEEPS:
        sub = df[df["sweep"] == sw]
        print(f"    {SWEEP_XLABELS[sw]}")
        for m in METHODS:
            if sub[m].isna().all():
                continue
            cells = []
            for x in sorted(sub["x"].unique()):
                v = {s: sub[(sub["x"] == x) & (sub["split"] == s)][m].iloc[0] for s in SPLITS}
                cells.append(f"{x:>4}: {100 * v['nonh']:5.1f} / {100 * v['h']:4.1f}")
            print(f"      {METHOD_LABELS[m]:<24}" + "   ".join(cells))

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w, h = style.TEXT_WIDTH, FIG_HEIGHT_IN
    panel_in = (w - YLABEL_IN - 2 * GUTTER_IN - RIGHT_PAD_IN) / 3
    fig = plt.figure(figsize=(w, h))

    def line_kw(mi, sp):
        return dict(color=style_for(mi), linestyle=SPLIT_LINE[sp],
                    marker=style.MARKERS[mi], markersize=3.4, markeredgewidth=0.6,
                    markerfacecolor=style_for(mi) if sp == "nonh" else "white",
                    markeredgecolor=style_for(mi), linewidth=1.1)

    for ci, sw in enumerate(SWEEPS):
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        pos = np.arange(len(xs))
        left = YLABEL_IN + ci * (panel_in + GUTTER_IN)
        ax = fig.add_axes([left / w, XTICKS_IN / h, panel_in / w, BODY_IN / h])
        for mi, m in enumerate(METHODS):
            if sub[m].isna().all():
                continue   # (c) has no action-only arm; (a)/(b) no single-view arm
            for sp in SPLITS:
                y = [sub[(sub["x"] == x) & (sub["split"] == sp)][m].iloc[0] for x in xs]
                ax.plot(pos, 100 * np.asarray(y, float), zorder=3, **line_kw(mi, sp))
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel(SWEEP_XLABELS[sw], labelpad=1.5)
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.minorticks_off()
        if ci == 0:
            ax.set_ylabel("Success Rate [%]", labelpad=2)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Two legend rows, one per factor. The method row shows filled markers on a
    # solid line (the in-distribution look); the split row is drawn in neutral
    # ink so it reads as a style key, not a fourth method.
    method_handles = [Line2D([], [], label=METHOD_LABELS[m], **line_kw(mi, "nonh"))
                      for mi, m in enumerate(METHODS)]
    split_handles = [Line2D([], [], color=style.INK_MUTED, linestyle=SPLIT_LINE[sp],
                            marker="o", markersize=3.4, markeredgewidth=0.6,
                            markerfacecolor=style.INK_MUTED if sp == "nonh" else "white",
                            markeredgecolor=style.INK_MUTED, linewidth=1.1, label=SPLIT_LABELS[sp])
                     for sp in SPLITS[::-1]]
    kw = dict(loc="upper center", frameon=False, borderaxespad=0, borderpad=0,
              handlelength=2.2, handletextpad=0.5, columnspacing=1.6)
    fig.legend(handles=method_handles, bbox_to_anchor=(0.5, 1.0), ncol=len(method_handles), **kw)
    fig.legend(handles=split_handles, bbox_to_anchor=(0.5, 1.0 - LEGEND_ROW_IN / h),
               ncol=len(split_handles), **kw)

    style.save(fig, OUT_NAME)


if __name__ == "__main__":
    main()
