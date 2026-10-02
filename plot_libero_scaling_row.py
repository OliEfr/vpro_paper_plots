r"""The three LIBERO scaling sweeps as one figure*, a plotted twin of the paper's
scaling table (tab:data_scaling_combined).

Reads ``results/libero_scaling_row.csv`` and draws a 2 x 3 grid: columns are the
three sweeps -- (a) action-labeled Panda episodes per task, (b) % of LIBERO-90
used as play data, (c) number of cross-embodiments in the video pretraining
mix -- and rows are the two task splits, held-out on top and in-distribution
below, the same order as the table. Run with no arguments:

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
video pretraining mix at all; and the cells at 5 episodes, 100% play data and
2 cross-embodiments are one and the same run.

LAYOUT. Each column shares its x axis between the two rows, so x tick labels and
the axis label are written once, under the bottom row. Each row shares its y
axis across the three columns (held-out 0-70, in-distribution 40-80), so the y
label and tick labels are written once, on the left column. The x positions are
categorical and evenly spaced -- the sweeps are not linear in their own units
(1/5/10/20 episodes, 1/2/4 embodiments) and the table reads them as steps.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style_science as style

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV = RESULTS_DIR / "libero_scaling_row.csv"
OUT_NAME = "libero_scaling_row_science"

METHODS = ["action_only_sr", "video_sr"]
METHOD_LABELS = {"action_only_sr": "Action-only", "video_sr": "+ LAM (ours)"}

SWEEPS = ["budget", "play", "xemb"]
SWEEP_XLABELS = {
    "budget": "(a) # Episodes with action labels",
    "play": "(b) % of LIBERO-90 as play data",
    "xemb": "(c) # X-embodiments in video pretraining",
}
SPLITS = ["h", "nonh"]   # top row first: same order as the table
SPLIT_YLABELS = {"h": "Held-out\nSR [%]", "nonh": "In-distribution\nSR [%]"}
SPLIT_YLIM = {"h": (0, 70), "nonh": (40, 82)}
SPLIT_YTICKS = {"h": [0, 20, 40, 60], "nonh": [40, 50, 60, 70, 80]}

# Bands in inches. Measured off the rendered text at 8pt, like the real-world
# row: two-line y labels plus tick labels on the left, one legend row on top,
# tick labels plus the one-line x label at the bottom.
YLABEL_IN = 0.42
GUTTER_IN = 0.22
RIGHT_PAD_IN = 0.08
LEGEND_ROW_IN = 0.13
LEGEND_GAP_IN = 0.05
ROW_GAP_IN = 0.08
XTICKS_IN = 0.34
BODY_IN = 0.78        # one row's plotting height
FIG_HEIGHT_IN = LEGEND_ROW_IN + LEGEND_GAP_IN + 2 * BODY_IN + ROW_GAP_IN + XTICKS_IN


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
            cells = []
            for x in sorted(sub["x"].unique()):
                v = {s: sub[(sub["x"] == x) & (sub["split"] == s)][m].iloc[0] for s in SPLITS}
                cells.append("  n/a" if pd.isna(v["nonh"]) else f"{x:>4}: {100 * v['nonh']:5.1f} / {100 * v['h']:4.1f}")
            print(f"      {METHOD_LABELS[m]:<14}" + "   ".join(cells))

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w, h = style.TEXT_WIDTH, FIG_HEIGHT_IN
    panel_in = (w - YLABEL_IN - 2 * GUTTER_IN - RIGHT_PAD_IN) / 3
    fig = plt.figure(figsize=(w, h))

    for ci, sw in enumerate(SWEEPS):
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        pos = np.arange(len(xs))
        left = YLABEL_IN + ci * (panel_in + GUTTER_IN)
        for ri, sp in enumerate(SPLITS):
            bottom = XTICKS_IN + (1 - ri) * (BODY_IN + ROW_GAP_IN)
            ax = fig.add_axes([left / w, bottom / h, panel_in / w, BODY_IN / h])
            for mi, m in enumerate(METHODS):
                y = [sub[(sub["x"] == x) & (sub["split"] == sp)][m].iloc[0] for x in xs]
                if all(pd.isna(v) for v in y):
                    continue   # (c): no action-only arm, the table's N/A
                ours = "(ours)" in METHOD_LABELS[m]
                ax.plot(pos, 100 * np.asarray(y, float), color=style_for(mi),
                        linestyle=style.LINE_OURS if ours else style.LINE_OTHER,
                        marker=style.MARKERS[mi], markersize=3.2, markeredgewidth=0.5,
                        markeredgecolor=style.MARKER_EDGE, linewidth=1.1, zorder=3)
            ax.set_xlim(-0.35, len(xs) - 0.65)
            ax.set_xticks(pos)
            ax.set_ylim(*SPLIT_YLIM[sp])
            ax.set_yticks(SPLIT_YTICKS[sp])
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            ax.minorticks_off()
            if ri == len(SPLITS) - 1:
                ax.set_xticklabels([str(int(x)) for x in xs])
                ax.set_xlabel(SWEEP_XLABELS[sw], labelpad=1.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if ci == 0:
                ax.set_ylabel(SPLIT_YLABELS[sp], labelpad=2)
            else:
                ax.tick_params(axis="y", labelleft=False)

    handles = [Line2D([], [], color=style_for(mi),
                      linestyle=style.LINE_OURS if "(ours)" in METHOD_LABELS[m] else style.LINE_OTHER,
                      marker=style.MARKERS[mi], markersize=3.2, markeredgewidth=0.5,
                      markeredgecolor=style.MARKER_EDGE, linewidth=1.1, label=METHOD_LABELS[m])
               for mi, m in enumerate(METHODS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, borderaxespad=0, borderpad=0, handlelength=2.2, handletextpad=0.5,
               columnspacing=1.6)

    style.save(fig, OUT_NAME)


if __name__ == "__main__":
    main()
