r"""The three LIBERO scaling sweeps as one figure*, a plotted twin of the paper's
scaling table (tab:data_scaling_combined).

Reads ``results/libero_scaling_row.csv`` and draws one row of three panels, one
per sweep -- (a) action-labeled Panda episodes per task, (b) % of LIBERO-90
used as play data, (c) number of cross-embodiments in the video pretraining
mix. Panels (a) and (b) draw, per method, the 40-task total as a line and the
two splits as a thin vertical range bar at each point: its top cap is the
in-distribution value, its bottom cap the held-out value, the mean marker sits
on it, and the two methods are dodged sideways so the bars do not overprint.
All three numbers are on the page with two lines per panel rather than six.
``--splits band`` draws the splits as a translucent band instead (same edges);
it is less busy still but the two methods' bands overlap into a blend over
most of the panel. NOT error bars -- the caption has to say so. Panel (c)
draws the total only, for the dual-view and the single-view
(sideview-only teacher, ``sideview_sr``) arm. The total is the 32/8
task-weighted mean of the two splits, derived here rather than stored, the way
libero_xemb_sweep's Total is. Run with no arguments:

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

TWO FACTORS, TWO CHANNELS (README.md). Method is colour plus marker shape; the
split is line versus range bar (the line is the total, the bar spans held-out
to in-distribution), and each factor gets its own legend row. Panels (a) and (b) share one
0-85 y axis, labelled once on the left; (c) is zoomed to 50-70 and carries its
own tick labels. The x positions are categorical and
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
# Split channel: the total is the line (solid, filled markers); the two stored
# splits are the edges of a band in the same colour -- in-distribution on top,
# held-out at the bottom. "total" is derived (see N_TASKS).
PANEL_BAND = {"budget": True, "play": True, "xemb": False}
BAND_ALPHA = 0.18
# --splits whisker: instead of a band, a thin vertical range bar per point from
# the held-out value (bottom cap) to the in-distribution value (top cap), the
# mean marker sitting on it; the two methods are dodged by +-DODGE so their
# bars do not overprint.
DODGE = 0.09
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
# (c) shows totals only, all in the 50s and 60s, so it gets its own zoomed y
# range -- and therefore its own tick labels, since it no longer shares the
# scale with (a) and (b). The point of the panel is the gap between the two
# arms, which 0-85 flattens to a sliver.
PANEL_YLIM = {"xemb": (50, 70)}
PANEL_YTICKS = {"xemb": [50, 55, 60, 65, 70]}

# Bands in inches. Measured off the rendered text at 8pt, like the real-world
# row: two-line y labels plus tick labels on the left, one legend row on top,
# tick labels plus the one-line x label at the bottom.
YLABEL_IN = 0.40
GUTTER_IN = 0.22
RIGHT_PAD_IN = 0.08
YTICKS_C_IN = 0.20    # (c)'s own tick labels ("50".."70"), taken out of its gutter
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", choices=["band", "whisker"], default="whisker",
                    help="how (a)/(b) show the two splits around the mean line")
    ap.add_argument("--out", default=OUT_NAME, help="output stem (default %(default)s)")
    args = ap.parse_args()
    df = load()

    print("\n  LIBERO scaling sweeps, success rate %  (all 40 tasks | in-distribution / held-out)")
    for sw in SWEEPS:
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        print(f"    {SWEEP_XLABELS[sw]}   (plotted: {'mean line + split band' if PANEL_BAND[sw] else 'mean line only'})")
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

    def line_kw(mi):
        return dict(color=style_for(mi), linestyle="-",
                    marker=style.MARKERS[mi], markersize=3.4, markeredgewidth=0.6,
                    markerfacecolor=style_for(mi), markeredgecolor=style_for(mi), linewidth=1.1)

    for ci, sw in enumerate(SWEEPS):
        sub = df[df["sweep"] == sw]
        xs = sorted(sub["x"].unique())
        pos = np.arange(len(xs))
        left = YLABEL_IN + ci * (panel_in + GUTTER_IN) + (YTICKS_C_IN if sw in PANEL_YLIM else 0.0)
        width_in = panel_in - (YTICKS_C_IN if sw in PANEL_YLIM else 0.0)
        ax = fig.add_axes([left / w, XTICKS_IN / h, width_in / w, BODY_IN / h])
        for mi, m in enumerate(METHODS):
            if sub[m].isna().all():
                continue   # (c) has no action-only arm; (a)/(b) no single-view arm
            px = pos + ((mi - 0.5) * 2 * DODGE if PANEL_BAND[sw] and args.splits == "whisker" else 0.0)
            y = 100 * np.asarray(series(sub, m, "total", xs), float)
            if PANEL_BAND[sw]:
                lo = 100 * np.asarray(series(sub, m, "h", xs), float)
                hi = 100 * np.asarray(series(sub, m, "nonh", xs), float)
                if args.splits == "band":
                    ax.fill_between(px, lo, hi, color=style_for(mi), alpha=BAND_ALPHA,
                                    linewidth=0, zorder=2)
                else:
                    ax.errorbar(px, y, yerr=[y - lo, hi - y], fmt="none", ecolor=style_for(mi),
                                elinewidth=0.7, capsize=2.2, capthick=0.7, alpha=0.85, zorder=2)
            ax.plot(px, y, zorder=3, **line_kw(mi))
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(int(x)) for x in xs])
        ax.set_xlabel(SWEEP_XLABELS[sw], labelpad=1.5)
        ax.set_ylim(*PANEL_YLIM.get(sw, YLIM))
        ax.set_yticks(PANEL_YTICKS.get(sw, YTICKS))
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.minorticks_off()
        if ci == 0:
            ax.set_ylabel("Success Rate [%]", labelpad=2)
        elif sw not in PANEL_YLIM:
            ax.tick_params(axis="y", labelleft=False)   # shares (a)'s scale

    # Two legend rows, one per factor. The method row shows the line; the split
    # row is drawn in neutral ink so it reads as a style key, not a fourth
    # method: the line is the 40-task mean, the band spans the two splits.
    from matplotlib.patches import Patch
    method_handles = [Line2D([], [], label=METHOD_LABELS[m], **line_kw(mi))
                      for mi, m in enumerate(METHODS)]
    split_handles = [
        Line2D([], [], color=style.INK_MUTED, linestyle="-", marker="o", markersize=3.4,
               markeredgewidth=0.6, markerfacecolor=style.INK_MUTED, markeredgecolor=style.INK_MUTED,
               linewidth=1.1, label="Mean over all tasks"),
        Patch(facecolor=style.INK_MUTED, alpha=0.3, linewidth=0,
              label="Held-out tasks (bottom edge) to in-distribution tasks (top edge)")
        if args.splits == "band" else
        Line2D([], [], color=style.INK_MUTED, linestyle="none", marker="$\\mathsf{I}$",
               markersize=7, markeredgewidth=0.4,
               label="Held-out tasks (bottom cap) to in-distribution tasks (top cap)"),
    ]
    kw = dict(loc="upper center", frameon=False, borderaxespad=0, borderpad=0,
              handlelength=2.2, handletextpad=0.5, columnspacing=1.6)
    fig.legend(handles=method_handles, bbox_to_anchor=(0.5, 1.0), ncol=len(method_handles), **kw)
    fig.legend(handles=split_handles, bbox_to_anchor=(0.5, 1.0 - LEGEND_ROW_IN / h),
               ncol=len(split_handles), **kw)

    style.save(fig, args.out)


if __name__ == "__main__":
    main()
