r"""The three LIBERO scaling sweeps as one figure*, a plotted twin of the paper's
scaling table (tab:data_scaling_combined).

Reads ``results/libero_scaling_row.csv`` and draws two rows. Columns (a) and
(b) are the episode-budget and the play-data sweep: their TOP panel is the mean
success rate over all 40 tasks, their BOTTOM panel the two task splits as
separate lines (in-distribution solid with filled markers, held-out dashed with
hollow markers). Column (c), the cross-embodiment sweep, spans both rows: it
draws the 40-task mean for the dual-view and the single-view (sideview-only
teacher, ``sideview_sr``) arm, with a thin horizontal rule between the two
curves -- halfway between the lowest dual-view and the highest single-view
point, so it exists only while one arm is above the other everywhere, which is
the point of the panel (the script fails if the curves cross). The mean is the 32/8
task-weighted mean of the two stored splits, derived here rather than stored,
the way libero_xemb_sweep's Total is. Run with no arguments:

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
teacher at the same recipe; the cells at 5 episodes, 100% play data and 2
cross-embodiments (dual-view) are one and the same run; and (c) is on its own
zoomed y axis.

TWO FACTORS, TWO CHANNELS (README.md). Method is colour plus marker shape; the
split is line style plus marker fill plus weight: the mean is the solid, opaque
line, the splits are dotted (in-distribution, filled) and dashed (held-out,
hollow) and drawn at SPLIT_ALPHA so the mean stays dominant without the colour
changing. One legend row carries both factors. The mean row and the split row
each share one y axis across (a) and (b), labelled once on the left with
every second tick labelled; (c) carries its own tick labels. The x positions
are categorical and evenly spaced -- the sweeps are not linear in their own
units (1/5/10/20 episodes, 1/2/4 embodiments) and the table reads them as
steps.
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

SWEEPS = ["budget", "play", "xemb"]
SWEEP_XLABELS = {
    "budget": "(a) # Episodes with action labels",
    "play": "(b) % of LIBERO-90 as play data",
    "xemb": "(c) # X-embodiments in video pretraining",
}
SPLIT_COLS = ["budget", "play"]        # the two-row columns
SPAN_COL = "xemb"                      # spans both rows, mean only

# Split channel (bottom row): in-distribution dotted / filled, held-out dashed /
# hollow, both at SPLIT_ALPHA; only the mean (top row and (c)) is the plain
# solid, opaque line.
SPLITS = ["nonh", "h"]
SPLIT_LINE = {"mean": "-", "nonh": (0, (1.2, 1.4)), "h": "--"}
SPLIT_FILLED = {"nonh": True, "h": False}
SPLIT_ALPHA = 0.55
SPLIT_LABELS = {"nonh": "In-distribution tasks", "h": "Held-out tasks"}
N_TASKS = {"nonh": 32, "h": 8}

# Ticks every step, labels on every second one (set_yticks + a blank label).
MEAN_YLIM, MEAN_YTICKS = (30, 80), [30, 40, 50, 60, 70, 80]
SPLIT_YLIM, SPLIT_YTICKS = (0, 85), [0, 20, 40, 60, 80]
# (c) is all in the 50s and 60s; on the split axis the dual-vs-single gap would
# read as a sliver, and the gap is the panel's point.
SPAN_YLIM, SPAN_YTICKS = (50, 70), [50, 55, 60, 65, 70]

# Bands in inches, measured off the rendered 8pt text like the real-world row.
YLABEL_IN = 0.40
YTICKS_C_IN = 0.20    # (c)'s own tick labels, taken out of its slot
GUTTER_IN = 0.22
RIGHT_PAD_IN = 0.08
LEGEND_ROW_IN = 0.13
LEGEND_ROWS = 1       # both factors in one row
LEGEND_GAP_IN = 0.05
ROW_GAP_IN = 0.08
XTICKS_IN = 0.34
BODY_IN = 0.62        # one row's plotting height
FIG_HEIGHT_IN = (LEGEND_ROWS * LEGEND_ROW_IN + LEGEND_GAP_IN
                 + 2 * BODY_IN + ROW_GAP_IN + XTICKS_IN)


def load():
    df = pd.read_csv(CSV, comment="#")
    assert set(df["sweep"]) == set(SWEEPS), sorted(set(df["sweep"]))
    assert set(df["split"]) == set(SPLITS), sorted(set(df["split"]))
    return df


def style_for(i):
    return style.PALETTE[i % len(style.PALETTE)]


def series(sub, m, sp, xs):
    """Values (fraction) of method m over xs for split sp; 'mean' is the
    task-weighted mean of the two stored splits."""
    at = lambda x, s: sub[(sub["x"] == x) & (sub["split"] == s)][m].iloc[0]
    if sp == "mean":
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
        print(f"    {SWEEP_XLABELS[sw]}   (plotted: {'mean only' if sw == SPAN_COL else 'mean + both splits'})")
        for m in METHODS:
            if sub[m].isna().all():
                continue
            t, i, h = (series(sub, m, sp, xs) for sp in ("mean", "nonh", "h"))
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
                    markerfacecolor=style_for(mi) if SPLIT_FILLED.get(sp, True) else "white",
                    markeredgecolor=style_for(mi), linewidth=1.1,
                    alpha=1.0 if sp == "mean" else SPLIT_ALPHA)

    def axis_common(ax, xs, ylim, yticks):
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_xticks(np.arange(len(xs)))
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(t) if k % 2 == 0 else "" for k, t in enumerate(yticks)])
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.minorticks_off()

    bottoms = {"top": XTICKS_IN + BODY_IN + ROW_GAP_IN, "bottom": XTICKS_IN}
    for ci, sw in enumerate(SPLIT_COLS):
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        pos = np.arange(len(xs))
        left = YLABEL_IN + ci * (panel_in + GUTTER_IN)
        for row in ("top", "bottom"):
            ax = fig.add_axes([left / w, bottoms[row] / h, panel_in / w, BODY_IN / h])
            for mi, m in enumerate(METHODS):
                if sub[m].isna().all():
                    continue
                for sp in (["mean"] if row == "top" else SPLITS):
                    y = 100 * np.asarray(series(sub, m, sp, xs), float)
                    ax.plot(pos, y, zorder=3, **line_kw(mi, sp))
            axis_common(ax, xs, *((MEAN_YLIM, MEAN_YTICKS) if row == "top" else (SPLIT_YLIM, SPLIT_YTICKS)))
            if row == "bottom":
                ax.set_xticklabels([str(int(x)) for x in xs])
                ax.set_xlabel(SWEEP_XLABELS[sw], labelpad=1.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if ci == 0:
                ax.set_ylabel("Mean SR [%]" if row == "top" else "Split SR [%]", labelpad=2)
            else:
                ax.tick_params(axis="y", labelleft=False)

    # (c): one tall panel, mean only, dual-view vs single-view, tied at each x.
    sub = df[df["sweep"] == SPAN_COL]
    xs = sorted(sub["x"].unique())
    pos = np.arange(len(xs))
    left = YLABEL_IN + 2 * (panel_in + GUTTER_IN) + YTICKS_C_IN
    ax = fig.add_axes([left / w, XTICKS_IN / h, (panel_in - YTICKS_C_IN) / w,
                       (2 * BODY_IN + ROW_GAP_IN) / h])
    ys = {}
    for mi, m in enumerate(METHODS):
        if sub[m].isna().all():
            continue
        ys[m] = 100 * np.asarray(series(sub, m, "mean", xs), float)
    # The separator: a thin muted rule halfway between the lowest dual-view and
    # the highest single-view mean. It only makes sense while the curves do not
    # cross, so refuse to draw a lie if they ever do.
    lo_dual, hi_single = ys["video_sr"].min(), ys["sideview_sr"].max()
    if lo_dual <= hi_single:
        raise SystemExit(f"(c): dual-view ({lo_dual:.1f}) no longer above single-view "
                         f"({hi_single:.1f}) everywhere -- drop the separator rule")
    ax.axhline((lo_dual + hi_single) / 2, color=style.INK_MUTED, linestyle="-",
               linewidth=0.6, zorder=2)   # linestyle explicit: the style's prop cycle would dash it
    print(f"    (c) separator rule at {(lo_dual + hi_single) / 2:.2f}% "
          f"(dual-view min {lo_dual:.1f}, single-view max {hi_single:.1f})")
    for mi, m in enumerate(METHODS):
        if m in ys:
            ax.plot(pos, ys[m], zorder=3, **line_kw(mi, "mean"))
    axis_common(ax, xs, SPAN_YLIM, SPAN_YTICKS)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel(SWEEP_XLABELS[SPAN_COL], labelpad=1.5)
    ax.set_ylabel("Mean SR [%]", labelpad=2)

    # One legend row, both factors: the three methods (as their mean line), then
    # the two split styles in neutral ink so they read as a key, not as methods.
    method_handles = [Line2D([], [], label=METHOD_LABELS[m], **line_kw(mi, "mean"))
                      for mi, m in enumerate(METHODS)]
    split_handles = [Line2D([], [], color=style.INK_MUTED, linestyle=SPLIT_LINE[sp],
                            marker="o", markersize=3.4, markeredgewidth=0.6,
                            markerfacecolor=style.INK_MUTED if SPLIT_FILLED[sp] else "white",
                            markeredgecolor=style.INK_MUTED, linewidth=1.1, alpha=SPLIT_ALPHA,
                            label=SPLIT_LABELS[sp])
                     for sp in SPLITS]
    handles = method_handles + split_handles
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, borderaxespad=0, borderpad=0, handlelength=2.2, handletextpad=0.5,
               columnspacing=1.4)

    style.save(fig, OUT_NAME)


if __name__ == "__main__":
    main()
