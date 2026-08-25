r"""Real-world success rate across tasks, at two demo budgets.

Reads ``results/realworld_alltasks.csv`` and emits
``figures/realworld_alltasks.pdf`` -- every script here names its output after
the CSV stem, so an input and its figure always share a name. Run with no
arguments to rebuild:

    python plot_realworld_bars.py

Encoding: x groups by task, **color** identifies the method, **hatch**
identifies the demo budget. Two channels for two factors, so a reader can hold
one fixed and scan the other.

Bars are ordered budget-major, so within a task each budget's methods stay
contiguous: the reader sees one 0-demo block and one 5-demo block, and the
Action-Only -> ours gap inside each. That gap is the figure's claim, so it is
the one made adjacent. The cost is that a single method's 0 -> 5 step is no
longer contiguous, so reading how much a method gains from demos means skipping
over the other method's bar. Swap the two loops that fill `offsets` to get
method-major ordering, which keeps each method's budgets together instead.

EMBEDDING IN LATEX
------------------
Built at 7.140in = \textwidth (516pt, per IEEEtran journal mode), so this is a
figure*, NOT a plain figure, and it needs \usepackage{graphicx}. On IEEEtran a
figure* floats to the top of a page; \usepackage{stfloats} lets it sit at the
bottom instead.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/realworld_alltasks.pdf}
      \caption{Real-world success rate over 25 rollouts on six manipulation
      tasks, at two demonstration budgets. Video pretraining improves success
      on every task at both budgets.}
      \label{fig:realworld_alltasks}
    \end{figure*}

It was a single-column figure through four tasks. Six groups do not fit
\columnwidth -- at 0.58in per group the three-line tick labels collide -- so it
moved to \textwidth. If the task list shrinks back, move it back rather than
leaving a half-empty two-column float.

Do not rescale it -- see README.md.

Input schema (see results/README.md). One row per (task, demo budget):

    task,n_demos,<method>_sr,<method>_sr,...

Any column ending in ``_sr`` is a method, in file order. Success rates are
stored as fractions in [0, 1] and converted to percent here. Every task must
carry the same set of budgets, or the grouped bars would not line up.

This file and ``realworld_scaling.csv`` share a schema and overlap wherever they
cover the same task at the same budget. That is deliberate -- the two figures
report one experiment, so the script cross-checks the overlap on the `task`
column and warns rather than letting the paper contradict itself.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style
import tasks

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "realworld_alltasks.csv"
SCALING_CSV = RESULTS_DIR / "realworld_scaling.csv"

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _sr suffix stripped. Order here does not matter; the CSV column
# order fixes the plotting order and the color assignment.
METHOD_LABELS = {
    "action_only_sr": "Action-Only",
    "video_sr": "w/ Video (ours)",
}


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_sr") else method)


def is_ours(method):
    """The hero method, flagged by "(ours)" in its display label. Kept in one
    place so the red bar edge and the legend never drift."""
    return "(ours)" in label_for(method)


def load(csv_path):
    """Return (task ids, budgets, {(method, budget): values in %}, methods, is_dummy).

    Values are indexed by task, in `task_ids` order. Empty cells read as NaN, so a
    partially filled dump plots the cells it has instead of drawing zeros --
    which would be a real success rate here, not a gap.
    """
    # The dummy marker is a comment line, not a column, so it survives being
    # read back by anything else that consumes this CSV.
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")

    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_sr'")

    task_ids = list(dict.fromkeys(df["task"]))          # first-seen order
    budgets = sorted(df["n_demos"].unique())

    # Grouped bars only line up if the grid is complete; a missing (task,
    # budget) row would silently shift every bar to its right.
    missing = [(t, b) for t in task_ids for b in budgets
               if df[(df["task"] == t) & (df["n_demos"] == b)].empty]
    if missing:
        raise SystemExit(f"{csv_path.name}: missing row(s) for "
                         + ", ".join(f"{t}@{int(b)}" for t, b in missing))

    values = {}
    for m in methods:
        if df[m].max(skipna=True) > 1.0:
            raise SystemExit(
                f"{csv_path.name}: {m} exceeds 1.0 -- success rates are stored "
                f"as fractions in [0, 1] and converted to percent by this script")
        for b in budgets:
            rows = df[df["n_demos"] == b].set_index("task")
            values[(m, b)] = rows.loc[task_ids, m].to_numpy(dtype=float) * 100.0

    return task_ids, budgets, values, methods, is_dummy


def cross_check(csv_path, task_ids, budgets, values, methods):
    """Warn where the scaling CSV disagrees with this one.

    The two files overlap wherever they cover the same task at the same budget,
    and both feed figures in the same paper -- so a mismatch means one dump is
    stale and the paper would print two different numbers for one measurement.

    Matched on the shared `task` column rather than on a filename, so renaming
    either file cannot quietly switch the check off.
    """
    if not SCALING_CSV.exists() or SCALING_CSV.resolve() == csv_path.resolve():
        return
    other = pd.read_csv(SCALING_CSV, comment="#")
    for ti, task in enumerate(task_ids):
        rows = other[other["task"] == task].set_index("n_demos")
        for m in methods:
            if m not in rows.columns:
                continue
            for b in budgets:
                if b not in rows.index:
                    continue
                here, there = values[(m, b)][ti], float(rows.loc[b, m]) * 100.0
                if np.isnan(here) or np.isnan(there):
                    continue
                if not np.isclose(here, there, atol=1e-6):
                    print(f"  ! {task} {m} @{int(b)} demos: {csv_path.name} says "
                          f"{here:.4f}%, {SCALING_CSV.name} says {there:.4f}%")


def print_table(task_ids, budgets, values, methods):
    """Text table view -- identity is never carried by color alone, and this is
    also the fastest way to spot a stale dump.

    The figure deliberately shows no numbers; this is where they live."""
    head = f"{'task':<16}{'demos':>7}" + "".join(f"{label_for(m):>18}" for m in methods)
    print(f"\n  Real-world success rate, %")
    print("  " + head)
    print("  " + "-" * len(head))
    for ti, task in enumerate(task_ids):
        for b in budgets:
            cells = "".join(
                f"{values[(m, b)][ti]:>18.4f}"
                if not np.isnan(values[(m, b)][ti]) else f"{'--':>18}"
                for m in methods)
            # tasks.label, not tasks.wrapped -- the wrapped form carries a
            # newline that would tear the table row in half.
            print("  " + f"{tasks.label(task):<16}{int(b):>7}" + cells)


def build_legends(fig, methods, budgets, left):
    """Two legends, one per encoding channel, stacked above the axes.

    A single combined legend would need methods x budgets entries to say what
    two short lists say directly, and it would imply the two factors are one.

    Placement is explicit rather than via constrained_layout: two legends both
    asking for "outside upper center" are given the same slot and silently
    drawn on top of each other.
    """
    from matplotlib.patches import Patch

    method_handles = [
        Patch(facecolor=style.PALETTE[i % len(style.PALETTE)],
              edgecolor=style.MARKER_EDGE, linewidth=0.6, label=label_for(m))
        for i, m in enumerate(methods)
    ]
    # Neutral fill for the hatch legend: these entries are about the hatch only,
    # and coloring them would suggest a method pairing that is not there.
    budget_handles = [
        Patch(facecolor="white", edgecolor=style.MARKER_EDGE, linewidth=0.6,
              hatch=style.HATCHES[i % len(style.HATCHES)],
              label=f"{int(b)} robot eps.")
        for i, b in enumerate(budgets)
    ]

    # Row spacing in inches, converted -- see the margin note in main(). At a
    # fixed fraction the second row creeps into the first as the figure shortens.
    row = 0.185 / fig.get_figheight()
    for handles, y in ((method_handles, 1.0), (budget_handles, 1.0 - row)):
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, y),
                   ncol=len(handles), frameon=False, borderaxespad=0,
                   handlelength=1.4, handleheight=0.9)


def main():
    # Raw formatter: the default one reflows paragraphs, which would collapse
    # the LaTeX snippet above into an unusable single block.
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                   help=f"result CSV (default: {DEFAULT_CSV.name})")
    p.add_argument("--name", default=None,
                   help="output basename in figures/ (default: the CSV stem)")
    p.add_argument("--style", choices=["paper", "science"], default="paper",
                   help="paper: exact IEEEtran/Computer-Modern match (default). "
                        "science: the SciencePlots aesthetic (garrettj403/SciencePlots).")
    args = p.parse_args()

    # The style is a swappable module exposing the same names (PALETTE, HATCHES,
    # apply_style, save, ...). Rebinding the module-global `style` here means the
    # draw code, which looks it up at call time, picks up whichever was asked for
    # without any per-call plumbing.
    global style
    name = args.name or args.csv.stem
    if args.style == "science":
        import style_science
        style = style_science
        if args.name is None:  # keep the two styles' outputs side by side
            name += "_science"

    if not args.csv.exists():
        raise SystemExit(f"no such results file: {args.csv}")

    task_ids, budgets, values, methods, is_dummy = load(args.csv)
    print_table(task_ids, budgets, values, methods)
    cross_check(args.csv, task_ids, budgets, values, methods)

    style.apply_style()
    import matplotlib.pyplot as plt

    n_tasks, n_methods, n_budgets = len(task_ids), len(methods), len(budgets)
    x = np.arange(n_tasks, dtype=float)

    # Bar geometry. Budget-major: one cluster of n_methods bars per budget, with
    # a gap between clusters so the eye reads "two blocks" rather than "four
    # bars". Each budget's methods stay contiguous, so the like-for-like
    # comparison at one budget is adjacent and the two budgets compare as whole
    # blocks.
    span, cluster_gap = 0.84, 0.14
    w = (span - cluster_gap * (n_budgets - 1)) / (n_methods * n_budgets)
    offsets = {}
    edge = -span / 2
    for bi, b in enumerate(budgets):
        for mi, m in enumerate(methods):
            offsets[(m, b)] = edge + w / 2 + (bi * n_methods + mi) * w + bi * cluster_gap

    # Explicit margins, not constrained_layout: the two stacked legends are
    # placed by hand (see build_legends), so the space they need is reserved by
    # hand too.
    #
    # Reserved in inches and converted, not written as fractions directly:
    # subplots_adjust takes fractions of the figure, so a hardcoded fraction
    # silently rescales the legend and tick-label margins whenever the height
    # changes -- and the failure mode is the two legend rows sliding into each
    # other, which is easy to miss in a thumbnail.
    legend_in = 0.48   # two legend rows above the axes
    # STIX, used by the no-LaTeX SciencePlots style, has taller descenders than
    # Computer Modern. Give that variant more room without shrinking the axes.
    ticks_in = 0.62 if args.style == "science" else 0.46
    # Axes height set from the width rather than left at the single-column 1.06in:
    # across \textwidth that would draw a 7:1 letterbox, and the 25-point gaps
    # this figure is about would flatten out of readability.
    # fig_w, not w -- `w` above is the bar width in data units.
    fig_w, axes_in = style.TEXT_WIDTH, 1.50
    h = axes_in + legend_in + ticks_in
    # Side margins in inches like the vertical ones, not as figure fractions: a
    # fraction tuned at \columnwidth silently doubles the margin at \textwidth,
    # and the y label ends up floating half an inch off its own axis.
    left_in, right_in = 0.47, 0.05
    left = left_in / fig_w
    fig, ax = plt.subplots(figsize=(fig_w, h))
    fig.subplots_adjust(left=left, right=1 - right_in / fig_w,
                        bottom=ticks_in / h, top=1 - legend_in / h)

    for bi, b in enumerate(budgets):
        for mi, m in enumerate(methods):
            ax.bar(
                x + offsets[(m, b)], values[(m, b)], width=w,
                facecolor=style.PALETTE[mi % len(style.PALETTE)],
                hatch=style.HATCHES[bi % len(style.HATCHES)],
                # Near-black outline on every bar, no red edge on "ours" -- the
                # deliberate exception to the highlight the other figures use.
                # matplotlib draws the hatch in the edge color, so a red edge
                # here paints the whole fill red-on-green rather than outlining
                # it, and the hero method ends up shouting instead of reading.
                # probing.pdf needs the highlight to pick one shape out of four
                # in three colors; two bars and a two-entry legend do not.
                edgecolor=style.MARKER_EDGE,
                linewidth=0.6, zorder=3,
            )

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([tasks.wrapped(t) for t in task_ids])
    ax.set_axisbelow(True)
    ax.grid(axis="y")

    # Task separators, at the midpoints between groups. Budget-major ordering
    # puts two blocks in each task, so without a rule the eye can read the gap
    # *between* two tasks as another block boundary. Set in INK_MUTED at the
    # spine weight rather than the lighter GRID: the y grid is a reading aid
    # behind the data, this is structure, and at GRID it did not out-rank the
    # gaps it is meant to separate. Still recessive grey, never black -- it is a
    # rule, not data -- and it stops at the axes rather than running through the
    # tick labels below.
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color=style.INK_MUTED, linewidth=0.6, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Success Rate [%]")

    build_legends(fig, methods, budgets, left)
    style.save(fig, name)

    n_missing = sum(int(np.isnan(v).sum()) for v in values.values())
    if n_missing:
        print(f"\n  note: {n_missing} empty cell(s) not yet run; drawn as gaps")
    if is_dummy:
        print("\n  " + "!" * 66)
        print(f"  ! {args.csv.name} is flagged DUMMY DATA -- it still contains")
        print("  ! placeholders. Do not ship this figure in the paper.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
