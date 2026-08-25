r"""All three real-world results in one row, as a two-column figure*.

Packs the three figures that already stand alone -- per-task success
(``realworld_alltasks.csv``), multi-view versus single-view
(``realworld_multiview_vs_side.csv``) and the demo-budget sweep
(``realworld_scaling.csv``) -- into a single \textwidth float, so the real-robot
story occupies one figure in the paper instead of three. Run with no arguments:

    python plot_realworld_row.py

SCIENCE STYLE AND PDF ONLY
--------------------------
Unlike every other script here this one does not take ``--style`` and does not
write a PNG. It exists to be dropped into the paper, and the paper takes the
SciencePlots variant; a paper-style twin nobody includes is one more thing to
keep in sync for nothing. That also means ``build_figures.sh`` runs it once
rather than twice -- see the SCIENCE_ONLY list there.

If a paper-style version is ever wanted, add the ``--style`` switch back the way
plot_radar_row.py has it; the draw code below is already style-agnostic.

EMBEDDING IN LATEX
------------------
Built at 7.140in = \textwidth (516pt, per IEEEtran journal mode), so this is a
figure*, NOT a plain figure. On IEEEtran a figure* floats to the top of a page;
\usepackage{stfloats} (already in the preamble) lets it sit at the bottom.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/realworld_row_science.pdf}
      \caption{}
      \label{fig:realworld_row}
    \end{figure*}

The caption is left empty on purpose -- it is the author's to write. Three
things it has to carry, because the figure no longer says them itself: the
panels are unlettered, so refer to them as left / middle / right; the bracketed
tick tags are abbreviated, so expand them (obj: novel object, place: novel
placement, move: novel movement, bkg: novel background); and the middle panel's
Mean column is pooled over its four tasks, so say so -- an aggregate that reads
as a fifth task is the misreading the bold rule in front of it is there to
prevent. The middle and right panels are both at the five-robot-episode budget,
and the right panel sweeps task3.

Include it at \textwidth and nothing is rescaled; any other width scales the
8pt text off the page. See README.md.

PANEL WIDTHS are not equal. The left panel carries six task groups and the
middle one five (four tasks plus Mean), and a group needs about 0.40in for the
abbreviated labels, so the widths come from the column counts rather than a
three-way split -- the scaling panel is sized first and the two bar panels
split what is left at one shared pitch. Splitting evenly instead collides the
left panel's labels while leaving the scaling panel half empty.

NO PANEL TITLES. Each panel's legend sits directly above it and names its arms,
which is what identifies the panel; a title row on top of that was a second
header saying much the same thing.

THE SHARED Y AXIS is the reason the row fits at all. All three panels are
success rate in percent on 0-100 with the same gridlines, so both the axis
label and the tick labels are written once, on the left panel; the other two
keep their tick marks and gridlines but drop the numbers. That buys 0.4in,
which goes into the bar panels rather than into margins.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style_science as style
import tasks

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALLTASKS_CSV = RESULTS_DIR / "realworld_alltasks.csv"
MULTIVIEW_CSV = RESULTS_DIR / "realworld_multiview_vs_side.csv"
SCALING_CSV = RESULTS_DIR / "realworld_scaling.csv"

OUT_NAME = "realworld_row_science"

# Labels are re-declared rather than imported from the three single-figure
# scripts: importing them would run three argparse modules for their constants
# alone. The cross-check in main() is what keeps these honest -- it fails the
# build if a name here stops matching the script that owns the figure.
METHOD_LABELS = {
    "action_only_sr": "Action-Only",
    "video_sr": "w/ Human Videos (ours)",
}
VIEW_LABELS = {
    "multiview_sr": "Multi-View",
    "side_only_sr": "Single-view",
}

# Multi-View keeps the green the other two panels give the video arm -- it is
# the same idea, so it should not change colour mid-figure. Single-view is NOT
# the blue it has in its own standalone figure: blue is Action-Only in both
# neighbouring panels here, and one colour meaning two different arms inside a
# single float is worse than this figure disagreeing with the standalone one.
#
# The orange is redder than PALETTE[2]'s #FF9500, which reads as glowing yellow
# next to the green, without going as dark as the burnt tones that read brown.
#
# It carries NO grayscale margin: these two bars have no hatch, so hue is the
# only thing separating them, and #F07800 sits at 1.09:1 relative luminance
# against the green -- a grayscale photocopy merges them completely. That is a
# knowing exception to README.md's "Hue never carries identity alone", taken
# because the colour was chosen by eye on screen. If the figure ever needs to
# survive grayscale, the cheap fix is to put the hatch back on one of the two
# bars; darkening the orange to #CC6000 only buys 1.53:1.
VIEW_COLORS = {
    "multiview_sr": style.PALETTE[1],
    "side_only_sr": "#F07800",
}

# Panel (b) is drawn over the same task order as panel (a) rather than the CSV's
# own order. Two panels side by side whose x axes list the same four tasks in
# different orders is a misreading waiting to happen.
TASK_ORDER = ["task1", "task2", "task3", "task4", "task5", "task6"]

# The scaling panel is sized first; the two bar panels then split what is left
# at one shared tick pitch, and the middle panel gives MIDDLE_SHRINK of its
# share to the left one. The left panel holds six groups of four bars against
# the middle's four groups of two, so it is the one that runs out of room first.
SCALING_AXES_IN = 1.37   # 1.52 less 10%
# Was 0.10 before the middle panel gained its Mean column. A fifth column plus a
# 10% haircut puts that panel's pitch under the floor below, so the transfer
# shrank rather than the column being squeezed in.
MIDDLE_SHRINK = 0.03     # of the middle panel's equal-pitch width, to the left
# Measured, not guessed: the widest short-form label line is 0.353in ("[obj]" /
# "on plate"), and 0.05in of clearance keeps it off the group rules.
MIN_TICK_PITCH_IN = 0.403

YLABEL_IN = 0.24    # "Success Rate [%]", rotated, on the left panel only
YTICKS_IN = 0.20    # "100" and friends -- left panel only, see below
GUTTER_IN = 0.20
RIGHT_PAD_IN = 0.09  # half of the last x tick label on the scaling panel
# Measured off the rendered legend artist, not guessed: at 0.15 the two-row
# legends reached 0.055in past the axes top. Nothing showed, because no bar in
# this dump reaches 100% -- a taller bar later would have printed through the
# legend text.
LEGEND_ROW_IN = 0.17
LEGEND_ROWS = 2      # every panel reserves two, so the three axes stay level
LEGEND_GAP_IN = 0.06  # clear air between the legend and the axes top
BODY_IN = 1.04       # 1.15 less 10%
XTICKS_IN = 0.52     # three-line task labels (bar panels) and the two-line
                     # "# Robot Episodes / (Task: ...)" under the scaling panel


def load_alltasks():
    df = pd.read_csv(ALLTASKS_CSV, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    task_ids = [t for t in TASK_ORDER if t in set(df["task"])]
    budgets = sorted(df["n_demos"].unique())
    values = {}
    for m in methods:
        for b in budgets:
            rows = df[df["n_demos"] == b].set_index("task")
            values[(m, b)] = rows.loc[task_ids, m].to_numpy(dtype=float) * 100.0
    return task_ids, budgets, values, methods


def load_multiview():
    """Per-task values plus the pooled mean the panel's last column shows.

    The mean is computed here, not read: the CSV deliberately carries no
    `overall` row any more (see results/README.md), because a dumped aggregate
    is a number nothing re-checks. Deriving it from the task rows means it
    cannot disagree with the bars beside it.

    Rollout-weighted, so it stays the pooled success rate if some task is ever
    evaluated over a different number of rollouts. With the current dump every
    task is 25 rollouts, so it coincides with the plain mean of the four -- and
    it reproduces the deleted `overall` row exactly (46.0 / 21.0).
    """
    df = pd.read_csv(MULTIVIEW_CSV).set_index("task")
    task_ids = [t for t in TASK_ORDER if t in df.index]
    methods = [c for c in VIEW_LABELS if c in df.columns]
    values = {m: df.loc[task_ids, m].to_numpy(dtype=float) * 100.0 for m in methods}
    weights = df.loc[task_ids, "n_rollouts"].to_numpy(dtype=float)
    means = {m: float(np.average(values[m], weights=weights)) for m in methods}
    return task_ids, values, means, methods


def load_scaling():
    df = pd.read_csv(SCALING_CSV, comment="#").sort_values("n_demos")
    methods = [c for c in df.columns if c.endswith("_sr")]
    task = str(df["task"].iloc[0])
    values = {m: df[m].to_numpy(dtype=float) * 100.0 for m in methods}
    return task, df["n_demos"].to_numpy(dtype=float), values, methods


def check_labels_match_owners():
    """Fail if this file's labels have drifted from the scripts that own them.

    The three single-figure scripts remain the source of truth for what an arm
    is called; this row just redraws their data. A silent divergence would put
    one method under two names in one paper, which is the failure the shared
    tasks.py exists to prevent -- so it is checked rather than trusted.
    """
    import importlib

    for modname, mapping, attr in (
        ("plot_realworld_bars", METHOD_LABELS, "METHOD_LABELS"),
        ("plot_realworld", METHOD_LABELS, "METHOD_LABELS"),
        ("plot_realworld_multiview_vs_side", VIEW_LABELS, "METHOD_LABELS"),
    ):
        owner = getattr(importlib.import_module(modname), attr)
        for key, name in mapping.items():
            if key in owner and owner[key] != name:
                raise SystemExit(
                    f"{Path(__file__).name}: {key} is '{name}' here but "
                    f"'{owner[key]}' in {modname}.py -- one figure per name")


def style_for(i):
    return style.PALETTE[i % len(style.PALETTE)]


def budget_hatch(bi, n_budgets):
    """Hatch for budget index `bi`, lowest budget striped and highest plain.

    Reversed relative to HATCHES' own order, which starts at "" -- so the
    0-episode bars are the striped ones and the 5-episode bars are solid. One
    helper rather than the expression inlined twice, because the bars and the
    legend swatches have to agree and nothing catches it if they stop.
    """
    return style.HATCHES[(n_budgets - 1 - bi) % len(style.HATCHES)]


def panel_bars(ax, task_ids, budgets, values, methods):
    """Left panel: grouped bars, colour is the method, hatch is the demo budget."""
    x = np.arange(len(task_ids), dtype=float)
    span, cluster_gap = 0.84, 0.14
    w = (span - cluster_gap * (len(budgets) - 1)) / (len(methods) * len(budgets))
    edge = -span / 2
    for bi, b in enumerate(budgets):
        for mi, m in enumerate(methods):
            off = edge + w / 2 + (bi * len(methods) + mi) * w + bi * cluster_gap
            ax.bar(x + off, values[(m, b)], width=w,
                   facecolor=style_for(mi),
                   hatch=budget_hatch(bi, len(budgets)),
                   edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([tasks.wrapped(t, short=True) for t in task_ids])
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=style.INK_MUTED, linewidth=0.5, zorder=0)


def panel_views(ax, task_ids, values, means, methods):
    """Middle panel: grouped bars, one pair per task, then a pooled Mean column.

    The standalone figure runs these bars horizontally to fit long task names in
    one column. Here the names are already on the left panel's x axis in the
    same order, so matching its orientation lets a reader compare the two.

    The Mean column is an aggregate of the four to its left, not a fifth task,
    so it is fenced off by a rule at the axis's full weight rather than by the
    hairline that divides one task from the next. Its label is set bold for the
    same reason. Getting this wrong is how a summary gets read as a measurement.
    """
    import matplotlib as mpl

    n = len(task_ids)
    x = np.arange(n + 1, dtype=float)   # tasks, then the mean slot
    w = 0.38
    for mi, m in enumerate(methods):
        series = np.append(values[m], means[m])
        # No hatch on either bar: these two are told apart by colour alone, which
        # is why VIEW_COLORS picks the orange it does -- see the note there.
        ax.bar(x + (mi - 0.5) * w, series, width=w,
               facecolor=VIEW_COLORS[m],
               edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
        # Printed on the Mean bars only. Every other bar is read off the shared
        # gridlines; the two summary numbers are the ones a reader will want to
        # quote, and they are the ones that appear in no other figure.
        #
        # Black, not white: on this green and this orange black runs about
        # 8-10:1 against the fill where white is under 3:1.
        ax.text(x[n] + (mi - 0.5) * w, means[m] - 2.0, f"{means[m]:.0f}",
                ha="center", va="top", color=style.INK, fontweight="bold",
                fontsize=mpl.rcParams["font.size"] - 1.0, zorder=5)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([tasks.wrapped(t, short=True) for t in task_ids]
                       + [r"$\mathbf{Mean}$"])
    for xi in x[:n - 1]:
        ax.axvline(xi + 0.5, color=style.INK_MUTED, linewidth=0.5, zorder=0)
    ax.axvline(n - 0.5, color=style.INK, linewidth=1.1, zorder=4)


def panel_scaling(ax, n_demos, values, methods, task):
    """Right panel: one line per method against the demo budget, x linear.

    The task name under the axis is read out of the CSV rather than written in,
    for the same reason tasks.py exists: this panel plots whatever task the
    scaling dump holds, and a hand-typed name goes stale the moment that dump
    changes to a different task.
    """
    for mi, m in enumerate(methods):
        ours = "(ours)" in METHOD_LABELS.get(m, m)
        ax.plot(n_demos, values[m],
                color=style_for(mi),
                linestyle="-" if ours else "--",
                marker=style.MARKERS[mi % len(style.MARKERS)],
                markersize=3.0, markeredgewidth=0.5,
                markeredgecolor=style.MARKER_EDGE, linewidth=1.1, zorder=3)
    ax.set_xlim(-2, max(n_demos) + 2)
    ax.set_xticks([0, 10, 20, 50])
    ax.set_xlabel(f"# Robot Episodes\n(Task: {tasks.label(task)})", labelpad=1.5)


def panel_legend(fig, handles, x0, w, h, ncol):
    """Legend for one panel, in the band between its title and its axes.

    ncol is per-panel and not derived from the entry count, because what fits is
    a function of the label lengths and the panel width, not of how many entries
    there are. The scaling panel is the narrow one and carries the longest
    label, so its two entries stack; at ncol=2 the row was 0.9in wider than the
    panel and the second label printed cut off at the page edge. Every panel
    reserves LEGEND_ROWS rows either way, which is what keeps the three axes
    level -- with no panel titles, the legend IS each panel's header.
    """
    y = 1.0
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=((x0 + w / 2) / fig.get_figwidth(), y),
               ncol=ncol, frameon=False, borderaxespad=0,
               handlelength=1.3, handleheight=0.9,
               handletextpad=0.4, columnspacing=1.0,
               labelspacing=0.25)


def main():
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    check_labels_match_owners()

    a_tasks, budgets, a_values, a_methods = load_alltasks()
    b_tasks, b_values, b_means, b_methods = load_multiview()
    c_task, n_demos, c_values, c_methods = load_scaling()

    print("\n  Real-world row, success rate %")
    print(f"    left   {len(a_tasks)} tasks x {len(budgets)} budgets x "
          f"{len(a_methods)} methods")
    print(f"    middle {len(b_tasks)} tasks x {len(b_methods)} view configs")
    print(f"    right  {tasks.label(c_task)}, budgets "
          f"{', '.join(str(int(n)) for n in n_demos)}")
    # The Mean column is the one number in this figure that is computed here
    # rather than redrawn from a dump the standalone scripts already table, so
    # it is the one that has to be printed to be checkable at all.
    print("    middle Mean column (pooled over the four tasks):")
    for m in b_methods:
        print(f"      {VIEW_LABELS.get(m, m):<14}{b_means[m]:8.4f}")

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    w = style.TEXT_WIDTH
    h = LEGEND_ROWS * LEGEND_ROW_IN + LEGEND_GAP_IN + BODY_IN + XTICKS_IN

    # Widths, derived rather than tabulated. Only the left panel pays for the y
    # label and the y tick labels -- the middle and right panels are the same
    # 0-100 success-rate axis with the same gridlines, so repeating "0 25 50 75
    # 100" twice more spends 0.4in restating a scale the reader can already
    # read off the left panel across an aligned gridline.
    # n_b + 1 columns, not n_b: the Mean column takes a full slot like a task.
    n_a, n_cols_b = len(a_tasks), len(b_tasks) + 1
    free = (w - (YLABEL_IN + YTICKS_IN) - 2 * GUTTER_IN - RIGHT_PAD_IN
            - SCALING_AXES_IN)
    pitch = free / (n_a + n_cols_b)
    given = n_cols_b * pitch * MIDDLE_SHRINK
    axes_in = {"a": n_a * pitch + given,
               "b": n_cols_b * pitch - given,
               "c": SCALING_AXES_IN}
    # Checked on the middle panel: after the transfer it is the one with the
    # tighter pitch, so it is where the labels collide first.
    tightest = axes_in["b"] / n_cols_b
    if tightest < MIN_TICK_PITCH_IN:
        raise SystemExit(
            f"tick pitch would be {tightest:.3f}in, under the "
            f"{MIN_TICK_PITCH_IN}in the abbreviated task labels need -- narrow "
            f"SCALING_AXES_IN, lower MIDDLE_SHRINK, or drop a task")

    # Panel origins walked left to right in inches.
    lefts, x = {}, 0.0
    for key in ("a", "b", "c"):
        lefts[key] = x + (YLABEL_IN + YTICKS_IN if key == "a" else 0.0)
        x = lefts[key] + axes_in[key] + GUTTER_IN

    fig = plt.figure(figsize=(w, h))
    bottom = XTICKS_IN / h
    body = BODY_IN / h

    axes = {}
    for key in ("a", "b", "c"):
        ax = fig.add_axes([lefts[key] / w, bottom, axes_in[key] / w, body])
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 25))
        if key != "a":
            # Ticks stay, labels go: the tick marks and gridlines still mark the
            # same 25% steps, so the shared scale is still legible.
            ax.tick_params(axis="y", labelleft=False)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0)
        axes[key] = ax

    panel_bars(axes["a"], a_tasks, budgets, a_values, a_methods)
    panel_views(axes["b"], b_tasks, b_values, b_means, b_methods)
    panel_scaling(axes["c"], n_demos, c_values, c_methods, c_task)
    axes["c"].tick_params(axis="x", length=2)
    axes["a"].set_ylabel("Success Rate [%]")

    method_handles = [
        Patch(facecolor=style_for(i), edgecolor=style.MARKER_EDGE, linewidth=0.5,
              label=METHOD_LABELS.get(m, m))
        for i, m in enumerate(a_methods)
    ]
    budget_handles = [
        Patch(facecolor="white", edgecolor=style.MARKER_EDGE, linewidth=0.5,
              hatch=budget_hatch(i, len(budgets)),
              label=f"{int(b)} robot eps.")
        for i, b in enumerate(budgets)
    ]
    view_handles = [
        Patch(facecolor=VIEW_COLORS[m], edgecolor=style.MARKER_EDGE, linewidth=0.5,
              label=VIEW_LABELS.get(m, m))
        for m in b_methods
    ]
    line_handles = [
        Line2D([], [], color=style_for(i),
               linestyle="-" if "(ours)" in METHOD_LABELS.get(m, m) else "--",
               marker=style.MARKERS[i % len(style.MARKERS)], markersize=3.0,
               markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.5,
               linewidth=1.1, label=METHOD_LABELS.get(m, m))
        for i, m in enumerate(c_methods)
    ]

    panel_legend(fig, method_handles + budget_handles,
                 lefts["a"], axes_in["a"], h, ncol=2)
    panel_legend(fig, view_handles,
                 lefts["b"], axes_in["b"], h, ncol=2)
    panel_legend(fig, line_handles,
                 lefts["c"], axes_in["c"], h, ncol=1)

    # PDF only, written here rather than through style.save, which also emits a
    # PNG -- see the note at the top of this file.
    style.FIG_DIR.mkdir(exist_ok=True)
    out = style.FIG_DIR / f"{OUT_NAME}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(style.FIG_DIR.parent)}")


if __name__ == "__main__":
    main()
