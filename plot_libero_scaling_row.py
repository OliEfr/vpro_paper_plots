r"""The three LIBERO scaling sweeps as one figure*, a plotted twin of the paper's
scaling table (tab:data_scaling_combined).

Reads ``results/libero_scaling_row.csv`` and draws one row of three panels, one
per sweep -- (a) action-labeled Panda episodes per task, (b) % of LIBERO-90
used as play data, (c) number of cross-embodiments in the video pretraining
mix. Panels (a) and (b) draw three curves per method: the 40-task total (solid,
filled markers), the in-distribution split (dotted, filled) and the held-out
split (dashed, hollow). Panel (c) draws the total only, for the dual-view and
the single-view (sideview-only teacher, ``sideview_sr``) arm. The total is the
32/8 task-weighted mean of the two splits, derived here rather than stored, the
way libero_xemb_sweep's Total is. Run with no arguments:

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
so a reader can hold one fixed and scan the other. Total is the solid style so
that panel (c), which shows totals alone, reads with the same key. The three
panels share one 0-85 y axis, labelled once on the left. The x positions are categorical and
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
# Split channel: total solid / filled, in-distribution dotted / filled, held-out
# dashed / hollow. "total" is derived (see N_TASKS), the other two are stored.
SPLIT_LINE = {"total": "-", "nonh": (0, (1.2, 1.4)), "h": "--"}
SPLIT_FILLED = {"total": True, "nonh": True, "h": False}
SPLIT_LABELS = {"total": "All tasks", "nonh": "In-distribution tasks", "h": "Held-out tasks"}
PANEL_SPLITS = {"budget": ["total", "nonh", "h"], "play": ["total", "nonh", "h"], "xemb": ["total"]}
N_TASKS = {"nonh": 32, "h": 8}

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


def series(sub, m, sp, xs):
    """Values (fraction) of method m over xs for split sp; 'total' is the
    task-weighted mean of the two stored splits."""
    at = lambda x, s: sub[(sub["x"] == x) & (sub["split"] == s)][m].iloc[0]
    if sp == "total":
        n = sum(N_TASKS.values())
        return [sum(N_TASKS[s] * at(x, s) for s in N_TASKS) / n for x in xs]
    return [at(x, sp) for x in xs]


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = load()

    print("\n  LIBERO scaling sweeps, success rate %  (all 40 tasks | in-distribution / held-out)")
    for sw in SWEEPS:
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        print(f"    {SWEEP_XLABELS[sw]}   (plotted: {', '.join(SPLIT_LABELS[s] for s in PANEL_SPLITS[sw])})")
        for m in METHODS:
            if sub[m].isna().all():
                continue
            t, i, h = (series(sub, m, sp, xs) for sp in ("total", "nonh", "h"))
            print(f"      {METHOD_LABELS[m]:<24}" + "   ".join(
                f"{x:>4}: {100 * t[k]:5.1f} | {100 * i[k]:5.1f} / {100 * h[k]:4.1f}" for k, x in enumerate(xs)))

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    w, h = style.TEXT_WIDTH, FIG_HEIGHT_IN
    panel_in = (w - YLABEL_IN - 2 * GUTTER_IN - RIGHT_PAD_IN) / 3
    fig = plt.figure(figsize=(w, h))

    def line_kw(mi, sp):
        return dict(color=style_for(mi), linestyle=SPLIT_LINE[sp],
                    marker=style.MARKERS[mi], markersize=3.4, markeredgewidth=0.6,
                    markerfacecolor=style_for(mi) if SPLIT_FILLED[sp] else "white",
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
            for sp in PANEL_SPLITS[sw]:
                y = series(sub, m, sp, xs)
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
    # solid line (the total's look); the split row is drawn in neutral ink so
    # it reads as a style key, not a fourth method.
    method_handles = [Line2D([], [], label=METHOD_LABELS[m], **line_kw(mi, "total"))
                      for mi, m in enumerate(METHODS)]
    split_handles = [Line2D([], [], color=style.INK_MUTED, linestyle=SPLIT_LINE[sp],
                            marker="o", markersize=3.4, markeredgewidth=0.6,
                            markerfacecolor=style.INK_MUTED if SPLIT_FILLED[sp] else "white",
                            markeredgecolor=style.INK_MUTED, linewidth=1.1, label=SPLIT_LABELS[sp])
                     for sp in ("total", "nonh", "h")]
    kw = dict(loc="upper center", frameon=False, borderaxespad=0, borderpad=0,
              handlelength=2.2, handletextpad=0.5, columnspacing=1.6)
    fig.legend(handles=method_handles, bbox_to_anchor=(0.5, 1.0), ncol=len(method_handles), **kw)
    fig.legend(handles=split_handles, bbox_to_anchor=(0.5, 1.0 - LEGEND_ROW_IN / h),
               ncol=len(split_handles), **kw)

    style.save(fig, OUT_NAME)


if __name__ == "__main__":
    main()
