r"""All three real-world results in one row, as a two-column figure*.

Packs the three figures that already stand alone -- per-task success
(``realworld_alltasks.csv``), multi-view versus single-view
(``realworld_multiview_vs_side.csv``) and the demo-budget sweep
(``realworld_scaling.csv``) -- into a single \textwidth float, so the real-robot
story occupies one figure in the paper instead of three. Run with no arguments:

    python plot_realworld_row.py

SCIENCE STYLE ONLY
------------------
Unlike every other script here this one does not take ``--style``. It exists to
be dropped into the paper, and the paper takes the SciencePlots variant; a
paper-style twin nobody includes is one more thing to keep in sync for nothing.
That also means ``build_figures.sh`` runs it once rather than twice -- see the
SCIENCE_ONLY list there.

It does write the PNG alongside the PDF, the way every other script here does.
The PDF is still the only file the paper includes; the PNG is for the last step
of README.md's checklist, and this is the figure that needs it most -- the panel
widths are hand-set. The tick labels and the legends are now measured off the
rendered text and checked (see required_pitch_in and check_legends_fit), but
nothing checks a bar against the legend above it.

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
placement, move: novel movement, bkg: novel background); and both bar panels
end in a Mean column pooled over the tasks to its left -- six on the left, four
in the middle -- so say so, because an aggregate that reads as one more task is
the misreading the bold rule in front of it is there to prevent. The middle and
right panels are both at the five-robot-episode budget, and the right panel
sweeps task3.

Include it at \textwidth and nothing is rescaled; any other width scales the
8pt text off the page. See README.md.

PANEL WIDTHS are not equal, and neither are their pitches. The left panel
carries seven columns (six tasks plus Mean) and the middle one five (four tasks
plus Mean), but the left panel also carries the widest label on either axis --
task5's "milk right" line, 0.474in against 0.353in for the widest the middle
panel draws -- so it needs the wider pitch as well as the greater number of
columns. Both are sized to leave TICK_LABEL_AIR_IN between neighbouring labels
and the scaling panel takes what is left, which is what makes it the panel that
shrinks when either bar panel needs more. Splitting evenly instead collides the
left panel's labels while leaving the scaling panel half empty.

THE RIGHT END of the row is set by the scaling panel's LEGEND, not by its axes.
That legend stacks one entry per row (see panel_legend) and its longest label is
"w/ Human Videos (ours)", so at 1.378in it is wider than the 1.07in axes it
heads and overhangs them on both sides. RIGHT_PAD_IN covers that overhang, which
is why it is bigger than the half tick label it would otherwise need to be.

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

# The two right-hand panels are sized outright and the left panel takes the
# rest, rather than everything being derived from one shared pitch: these two
# are the ones that get adjusted by eye, and stating their widths directly beats
# expressing each adjustment as a fraction of a fraction.
#
# Both moved when the left panel took on a Mean column: that column costs it
# 0.80 of a task slot, which at the old widths dropped its pitch from 0.486in
# to 0.429in and overlapped "milk right" into "in bowl" on either side of it.
# The two bar panels are now sized to clear required_pitch_in() by about the
# same margin -- 0.459in of pitch on the left against the 0.455in its widest
# neighbouring pair needs, 0.404in on the middle against the 0.399in its Mean
# label needs -- and the scaling panel paid for both. It is the panel that can:
# it plots five points on a monotone pair of lines, where the bar panels are
# carrying eleven three-line tick labels between them.
SCALING_AXES_IN = 1.07   # 1.37 less the 0.30 the left panel's Mean column cost
MIDDLE_AXES_IN = 1.94    # 1.925, rounded up to keep clear of its own floor

# Both bar panels end in a Mean column, and it gets a NARROWER slot than a task
# column -- which is what makes the middle panel fit at its width. Its label is
# one short bold word (0.283in) against a task's three stacked lines (0.353in),
# so an equal share would spend the panel's scarcest inches on its least
# demanding column. At an equal share the four task labels drop to a 0.385in
# pitch and "on plate" under task1 runs into "on plate" under task2 -- they do
# not technically overlap, but they read as one phrase, which is worse than a
# collision because nothing looks broken.
#
# The slot scales the BARS INSIDE IT by the same factor rather than dropping
# full-width bars into a short slot. A task column spends 84% (left panel) or
# 76% (middle) of its width on bars and the rest on the air that separates one
# column from the next; at full width in a 0.80 slot the middle panel's pair
# came within 0.008 data units of the fence rule on one side and the right spine
# on the other, so the Mean column read as wedged in rather than set apart.
# Scaled, it keeps a task column's proportions at 80% of its size.
MEAN_SLOT_UNITS = 0.80   # of one task column, bars and surrounding air alike
MEAN_TICK_LABEL = r"$\mathbf{Mean}$"

# Clear air between one x tick label and its neighbour. The pitch each bar panel
# needs is measured from this and the rendered labels rather than stated -- see
# required_pitch_in. 0.042in is not a new judgement: it is what the middle panel
# already shipped at, and the 0.395in floor it replaces implied the same thing
# against the label it was measured from. Roughly two thirds of an 8pt
# inter-word space, and that is the point -- go under it and two labels stop
# looking like two labels and start reading as one phrase, which is worse than
# an outright collision because nothing looks broken.
TICK_LABEL_AIR_IN = 0.042

YLABEL_IN = 0.24    # "Success Rate [%]", rotated, on the left panel only
YTICKS_IN = 0.20    # "100" and friends -- left panel only, see below
GUTTER_IN = 0.20
# Half of the scaling panel's last x tick label is only 0.045in; this is set by
# that panel's legend instead, which at 1.378in overhangs the 1.07in axes it is
# centred on by 0.154in a side and would print off the page edge without it.
# check_legends_fit() is the guard.
RIGHT_PAD_IN = 0.17
# Measured off the rendered legend artist, not guessed: at 0.15 the two-row
# legends reached 0.055in past the axes top. Nothing showed, because no bar in
# this dump reaches 100% -- a taller bar later would have printed through the
# legend text.
LEGEND_ROW_IN = 0.17
LEGEND_ROWS = 2      # every panel reserves two, so the three axes stay level
LEGEND_GAP_IN = 0.06  # clear air between the legend and the axes top
XTICKS_IN = 0.52     # three-line task labels (bar panels) and the two-line
                     # "# Robot Episodes / (Task: ...)" under the scaling panel

# The overall height is the knob, and the plot body is what is left after the
# text bands -- not the other way round. The legend rows and the tick-label band
# are 8pt type; they do not scale with the figure, so every inch taken off the
# total comes out of the body. This 10% cut off the whole figure (1.96 -> 1.76)
# is therefore a 19% cut to the plotting area.
FIG_HEIGHT_IN = 1.76
MIN_BODY_IN = 0.70   # below this the 0-100 axis stops being readable


def load_alltasks():
    """Per-task values plus the pooled mean the panel's last column shows.

    One mean per (method, budget) -- the panel's Mean column carries the same
    four bars every task column does, so the aggregate is taken along tasks
    only. Rollout-weighted and derived here rather than dumped, for the reason
    load_multiview() gives: a dumped aggregate is a number nothing re-checks.
    """
    df = pd.read_csv(ALLTASKS_CSV, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    task_ids = [t for t in TASK_ORDER if t in set(df["task"])]
    budgets = sorted(df["n_demos"].unique())
    values, means = {}, {}
    for m in methods:
        for b in budgets:
            rows = df[df["n_demos"] == b].set_index("task")
            values[(m, b)] = rows.loc[task_ids, m].to_numpy(dtype=float) * 100.0
            weights = rows.loc[task_ids, "n_rollouts"].to_numpy(dtype=float)
            means[(m, b)] = float(np.average(values[(m, b)], weights=weights))
    return task_ids, budgets, values, means, methods


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


def _line_widths_in(fig, text):
    """Width of each line of `text` in inches, as the active style renders it.

    Measured rather than tabulated because every number it feeds is a property
    of the font: a constant here would be right for the style and the freetype
    build it was measured on and quietly wrong on the next one.
    """
    renderer = fig.canvas.get_renderer()
    widths = []
    for line in text.split("\n"):
        artist = fig.text(0.5, 0.5, line)
        widths.append(artist.get_window_extent(renderer=renderer).width / fig.dpi)
        artist.remove()
    return widths


def required_pitch_in(fig, task_ids):
    """Column pitch this panel's x tick labels need, from the labels themselves.

    The 0.395in constant this replaces was wrong twice over. It was one number
    for two panels that do not draw the same labels, and it was measured off
    "[obj]" / "on plate" at 0.353in, missing "milk right" -- the widest line on
    the left panel's axis at 0.474in. Adding the Mean column dropped that panel
    to a 0.429in pitch, where "milk right" overlapped "in bowl" on both sides of
    it by 0.045in, and the guard passed.

    What binds is the widest ADJACENT PAIR, not the widest label: the labels are
    centred on their columns, so what has to fit between two centres is half of
    each neighbour plus TICK_LABEL_AIR_IN.
    """
    lines = [_line_widths_in(fig, tasks.wrapped(t, short=True)) for t in task_ids]
    # tasks.wrapped() gives every label the same number of lines and they are
    # set from a common top, so line i only ever meets line i of its neighbour.
    need = max(((a + b) / 2 + TICK_LABEL_AIR_IN)
               for left, right in zip(lines, lines[1:])
               for a, b in zip(left, right))
    # The Mean label against the last task's, at the closer spacing the narrower
    # slot puts them at. Checked against that neighbour's widest line rather
    # than line by line: it is one mathtext line, and mathtext sets its own
    # height, so which of the three rows it lands level with is not ours to say.
    mean_w = max(_line_widths_in(fig, MEAN_TICK_LABEL))
    mean_gap_units = (1.0 + MEAN_SLOT_UNITS) / 2
    return max(need, ((mean_w + max(lines[-1])) / 2 + TICK_LABEL_AIR_IN)
               / mean_gap_units)


def check_legends_fit(fig, legends):
    """Fail if a panel legend prints off the page.

    With no panel titles the legend is each panel's header, and it is centred on
    its panel rather than fitted to it -- the scaling panel's is wider than its
    axes. That makes the row's outer edges a function of the legend text, so a
    renamed arm can push one off the canvas. Nothing else here would notice:
    a fig.legend is not clipped to anything, it just goes missing at the crop.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    limit = fig.get_figwidth() * fig.dpi
    for key, legend in legends.items():
        box = legend.get_window_extent(renderer=renderer)
        over = max(-box.x0, box.x1 - limit) / fig.dpi
        if over > 0:
            raise SystemExit(
                f"panel '{key}' legend runs {over:.3f}in off the figure -- it is "
                f"{box.width / fig.dpi:.3f}in wide and centred on a "
                f"{key} panel that cannot hold it; widen RIGHT_PAD_IN, shorten "
                f"the labels, or give the panel a different ncol")


def mean_slot_x(n):
    """Centre of the Mean slot in a bar panel with `n` task columns.

    Task slots are one data unit each and so end at ``n - 0.5``; the Mean slot
    starts there and is MEAN_SLOT_UNITS wide, with its centre in the middle of
    it -- the narrower column is a narrower slot, not a full slot with the bars
    pushed to one side.
    """
    return n - 0.5 + MEAN_SLOT_UNITS / 2


def finish_bar_panel(ax, task_ids):
    """Column dividers, the Mean fence, x limits and tick labels.

    Shared by both bar panels. They differ in what goes inside a column, not in
    how the columns are fenced, and the Mean fence in particular has to be drawn
    identically in both -- it is the mark that says "aggregate, not a task", so
    a reader who learns it in one panel should be able to read it in the other.
    """
    n = len(task_ids)
    x = np.arange(n, dtype=float)
    for xi in x[:n - 1]:
        ax.axvline(xi + 0.5, color=style.INK_MUTED, linewidth=0.5, zorder=0)
    # The Mean column aggregates the columns to its left instead of adding one
    # more measurement to them, so it is fenced off by a rule at the axis's full
    # weight rather than by the hairline that divides one task from the next,
    # and its label is set bold for the same reason. Getting this wrong is how a
    # summary gets read as a measurement.
    ax.axvline(n - 0.5, color=style.INK, linewidth=1.1, zorder=4)
    # Right edge is the end of the Mean slot, not half a task-slot past its
    # centre -- otherwise the narrower slot buys nothing back.
    ax.set_xlim(-0.5, n - 0.5 + MEAN_SLOT_UNITS)
    ax.set_xticks(np.append(x, mean_slot_x(n)))
    ax.set_xticklabels([tasks.wrapped(t, short=True) for t in task_ids]
                       + [MEAN_TICK_LABEL])


def panel_bars(ax, task_ids, budgets, values, means, methods):
    """Left panel: grouped bars, colour is the method, hatch is the demo budget.

    Ends in the same pooled Mean column the middle panel carries, fenced the
    same way -- see finish_bar_panel.

    Its four Mean values are printed like the middle panel's two, but ROTATED
    and set above the bar rather than horizontally inside it. Four bars to a
    cluster puts one at 0.064in wide here, against 0.093in for a horizontal
    two-digit number at the middle panel's 7pt -- it would be wider than the bar
    it labels. Turned on its side the number's CAP HEIGHT is what has to fit the
    bar instead of its length, and digits carry neither ascender nor descender,
    so 6pt bold clears the bar by 0.006in where 7pt overruns it by 0.004in.

    The length is then charged to headroom instead, which this column has: the
    tallest Mean bar is 42%, and 0.079in of number above it reaches 52%. Note
    that is a fact about this dump, not about the layout -- a Mean above roughly
    88% would run the number into the legend, and nothing here checks that.
    """
    import matplotlib as mpl

    n = len(task_ids)
    x = np.arange(n, dtype=float)
    span, cluster_gap = 0.84, 0.14
    w = (span - cluster_gap * (len(budgets) - 1)) / (len(methods) * len(budgets))
    edge = -span / 2
    mean_x = mean_slot_x(n)
    widths = np.append(np.full(n, w), w * MEAN_SLOT_UNITS)
    for bi, b in enumerate(budgets):
        for mi, m in enumerate(methods):
            # The offset within the cluster is scaled along with the bar width,
            # so the four Mean bars stay a centred, uncut copy of a task
            # cluster rather than a full-width one overflowing the short slot.
            off = edge + w / 2 + (bi * len(methods) + mi) * w + bi * cluster_gap
            ax.bar(np.append(x + off, mean_x + off * MEAN_SLOT_UNITS),
                   np.append(values[(m, b)], means[(m, b)]), width=widths,
                   facecolor=style_for(mi),
                   hatch=budget_hatch(bi, len(budgets)),
                   edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
            # Black on white here rather than the middle panel's black on fill,
            # so there is no contrast argument to make -- but it is the same
            # bold, because it is the same kind of number.
            ax.text(mean_x + off * MEAN_SLOT_UNITS, means[(m, b)] + 2.0,
                    f"{means[(m, b)]:.0f}", rotation=90,
                    ha="center", va="bottom", color=style.INK, fontweight="bold",
                    fontsize=mpl.rcParams["font.size"] - 2.0, zorder=5)
    finish_bar_panel(ax, task_ids)


def panel_views(ax, task_ids, values, means, methods):
    """Middle panel: grouped bars, one pair per task, then a pooled Mean column.

    The standalone figure runs these bars horizontally to fit long task names in
    one column. Here the names are already on the left panel's x axis in the
    same order, so matching its orientation lets a reader compare the two.

    The Mean column is fenced off and labelled by finish_bar_panel, the same way
    the left panel's is.
    """
    import matplotlib as mpl

    n = len(task_ids)
    x = np.arange(n, dtype=float)
    mean_x = mean_slot_x(n)
    w = 0.38
    widths = np.append(np.full(n, w), w * MEAN_SLOT_UNITS)
    for mi, m in enumerate(methods):
        # The offset from the slot centre scales with the slot, like the width,
        # so the pair keeps a task column's air on both sides of it instead of
        # touching the fence rule on one side and the right spine on the other.
        off = (mi - 0.5) * w
        # No hatch on either bar: these two are told apart by colour alone, which
        # is why VIEW_COLORS picks the orange it does -- see the note there.
        ax.bar(np.append(x + off, mean_x + off * MEAN_SLOT_UNITS),
               np.append(values[m], means[m]), width=widths,
               facecolor=VIEW_COLORS[m],
               edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
        # Printed on the Mean bars only. Every other bar is read off the shared
        # gridlines; the two summary numbers are the ones a reader will want to
        # quote, and they are the ones that appear in no other figure.
        #
        # Black, not white: on this green and this orange black runs about
        # 8-10:1 against the fill where white is under 3:1.
        ax.text(mean_x + off * MEAN_SLOT_UNITS, means[m] - 2.0, f"{means[m]:.0f}",
                ha="center", va="top", color=style.INK, fontweight="bold",
                fontsize=mpl.rcParams["font.size"] - 1.0, zorder=5)
    finish_bar_panel(ax, task_ids)


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
    return fig.legend(handles=handles, loc="upper center",
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

    a_tasks, budgets, a_values, a_means, a_methods = load_alltasks()
    b_tasks, b_values, b_means, b_methods = load_multiview()
    c_task, n_demos, c_values, c_methods = load_scaling()

    print("\n  Real-world row, success rate %")
    print(f"    left   {len(a_tasks)} tasks x {len(budgets)} budgets x "
          f"{len(a_methods)} methods")
    print(f"    middle {len(b_tasks)} tasks x {len(b_methods)} view configs")
    print(f"    right  {tasks.label(c_task)}, budgets "
          f"{', '.join(str(int(n)) for n in n_demos)}")
    # The Mean columns are the numbers in this figure computed here rather than
    # redrawn from a dump the standalone scripts already table, so they are the
    # ones that have to be printed to be checkable at all.
    print(f"    left Mean column (pooled over the {len(a_tasks)} tasks):")
    for m in a_methods:
        for b in budgets:
            print(f"      {METHOD_LABELS.get(m, m):<24}"
                  f"{int(b)} robot eps.{a_means[(m, b)]:8.4f}")
    print(f"    middle Mean column (pooled over the {len(b_tasks)} tasks):")
    for m in b_methods:
        print(f"      {VIEW_LABELS.get(m, m):<14}{b_means[m]:8.4f}")

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    w = style.TEXT_WIDTH
    h = FIG_HEIGHT_IN
    body_in = h - (LEGEND_ROWS * LEGEND_ROW_IN + LEGEND_GAP_IN + XTICKS_IN)
    if body_in < MIN_BODY_IN:
        raise SystemExit(
            f"FIG_HEIGHT_IN={h}in leaves {body_in:.2f}in of plotting area, under "
            f"the {MIN_BODY_IN}in floor -- the legend and tick bands are fixed "
            f"8pt type and cannot absorb any of the cut")

    # Widths, derived rather than tabulated. Only the left panel pays for the y
    # label and the y tick labels -- the middle and right panels are the same
    # 0-100 success-rate axis with the same gridlines, so repeating "0 25 50 75
    # 100" twice more spends 0.4in restating a scale the reader can already
    # read off the left panel across an aligned gridline.
    # The two right-hand panels take their stated widths; the left panel takes
    # whatever is left, which is why it is the one that grows when either of the
    # others is trimmed.
    n_a = len(a_tasks)
    axes_in = {"a": (w - (YLABEL_IN + YTICKS_IN) - 2 * GUTTER_IN - RIGHT_PAD_IN
                     - MIDDLE_AXES_IN - SCALING_AXES_IN),
               "b": MIDDLE_AXES_IN,
               "c": SCALING_AXES_IN}

    # Panel origins walked left to right in inches.
    lefts, x = {}, 0.0
    for key in ("a", "b", "c"):
        lefts[key] = x + (YLABEL_IN + YTICKS_IN if key == "a" else 0.0)
        x = lefts[key] + axes_in[key] + GUTTER_IN

    fig = plt.figure(figsize=(w, h))

    # Both bar panels are checked, not just one: which of the two is tighter
    # depends on the widths above and on which task set it draws, and the loser
    # is where labels merge first. Checked here rather than beside the width
    # arithmetic because it needs a figure to measure the labels against.
    pitches = {"a": axes_in["a"] / (n_a + MEAN_SLOT_UNITS),
               "b": axes_in["b"] / (len(b_tasks) + MEAN_SLOT_UNITS)}
    for key, task_ids in (("a", a_tasks), ("b", b_tasks)):
        need = required_pitch_in(fig, task_ids)
        if pitches[key] < need:
            raise SystemExit(
                f"panel '{key}' tick pitch would be {pitches[key]:.3f}in against "
                f"the {need:.3f}in its labels need at {TICK_LABEL_AIR_IN}in of "
                f"air -- widen it by trimming MIDDLE_AXES_IN or SCALING_AXES_IN, "
                f"or shorten the labels")

    bottom = XTICKS_IN / h
    body = body_in / h

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

    panel_bars(axes["a"], a_tasks, budgets, a_values, a_means, a_methods)
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

    legends = {
        "a": panel_legend(fig, method_handles + budget_handles,
                          lefts["a"], axes_in["a"], h, ncol=2),
        "b": panel_legend(fig, view_handles,
                          lefts["b"], axes_in["b"], h, ncol=2),
        "c": panel_legend(fig, line_handles,
                          lefts["c"], axes_in["c"], h, ncol=1),
    }
    check_legends_fit(fig, legends)

    style.save(fig, OUT_NAME)


if __name__ == "__main__":
    main()
