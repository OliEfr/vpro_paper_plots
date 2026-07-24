r"""Real-world data efficiency: success rate against the number of robot demos.

Reads ``results/realworld_scaling.csv`` and emits
``figures/realworld_scaling.pdf`` -- every script here names its output after
the CSV stem, so an input and its figure always share a name. Run with no
arguments to rebuild:

    python plot_realworld.py

Encoding: x is the demo budget, y is success rate in percent, and **color +
dash + marker shape** together identify the method. One factor, so all three
channels are spent on it rather than split -- the opposite of probing.pdf,
which has two factors and gives each its own channel.

DUMMY DATA. ``results/realworld_scaling.csv`` currently holds placeholders at
10-decimal precision (0.1234567891 and friends), not measurements, so the
layout can be reviewed before the runs finish. The file is flagged with a
``# DUMMY DATA`` header line and the script prints a banner while it is there;
drop that line when the real numbers land.

EMBEDDING IN LATEX
------------------
Built at 3.487in = \columnwidth (252pc, per IEEEtran journal mode), so it is a
plain figure, not a figure*. Needs \usepackage{graphicx}.

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/realworld_scaling.pdf}
      \caption{Real-world data efficiency on \emph{Banana in bowl}. Policy
      success rate over 20 rollouts as a function of the number of teleoperated
      robot demonstrations. Video pretraining lifts the whole curve, and the gap
      is widest in the low-demo regime.}
      \label{fig:realworld_scaling}
    \end{figure}

Do not rescale it. Every label is set at 8pt (IEEEtran \footnotesize) in
Computer Modern, so at width=\columnwidth the labels land on the page at
exactly that size; any other width scales the text off it.

Input schema (see results/README.md). One row per demo budget:

    task,n_demos,<method>_sr,<method>_sr,...

Any column ending in ``_sr`` is treated as a method, in file order, so adding
a third method is a schema change only. Success rates are stored as fractions
in [0, 1] and converted to percent here -- the axis is the only place the
number is ever a percentage.

Unlike the probing dumps, 0 is a REAL value here: a policy trained on zero
demos genuinely scores 0% success, and drawing that as a gap would hide the
most important point on the curve. Not-yet-run cells are left empty instead.

X SPACING is linear, so the 20 -> 50 gap is drawn 6x wider than 0 -> 5. That is
the honest encoding of a quantitative axis and it is what makes the saturation
visible. For equal spacing between budgets instead, plot against
``np.arange(len(n_demos))`` and label the ticks with the budgets -- but then the
curve's flattening is an artifact of the axis, and the caption has to say so.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style
import tasks

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "realworld_scaling.csv"

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _sr suffix stripped. Order here does not matter; the CSV column
# order fixes the plotting order, the color and the marker assignment.
METHOD_LABELS = {
    "action_only_sr": "Action-Only",
    "video_sr": "w/ Video (ours)",
}


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_sr") else method)


def is_ours(method):
    """The hero method, flagged by "(ours)" in its display label. Kept in one
    place so the solid line, the red marker edge and the legend never drift."""
    return "(ours)" in label_for(method)


def load(csv_path):
    """Return (task, demo counts, {method: success rate in %}, methods, is_dummy).

    Empty cells read as NaN, which matplotlib skips -- so a partially filled
    dump plots the budgets it has instead of failing or drawing zeros.
    """
    # The dummy marker is a comment line, not a column, so it survives being
    # read back by anything else that consumes this CSV.
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")

    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_sr'")

    # The `task` column exists so this file and realworld_alltasks.csv share one
    # schema and the bar script can cross-check them by data rather than by
    # filename. This figure is one task's curve, though: two tasks would put
    # four lines on one axes, and each extra task doubles that.
    tasks = list(dict.fromkeys(df["task"]))
    if len(tasks) > 1:
        raise SystemExit(f"{csv_path.name}: holds {len(tasks)} tasks "
                         f"({', '.join(map(str, tasks))}); this figure plots one. "
                         f"Split it, or plot the tasks together with "
                         f"plot_realworld_bars.py")

    df = df.sort_values("n_demos")
    n_demos = df["n_demos"].to_numpy(dtype=float)

    values = {}
    for m in methods:
        sr = df[m].to_numpy(dtype=float)
        if np.nanmax(sr, initial=0.0) > 1.0:
            raise SystemExit(
                f"{csv_path.name}: {m} exceeds 1.0 -- success rates are stored "
                f"as fractions in [0, 1] and converted to percent by this script")
        values[m] = sr * 100.0
    return tasks[0], n_demos, values, methods, is_dummy


def print_table(task_label, n_demos, values, methods):
    """Text table view -- identity is never carried by color alone, and this is
    also the fastest way to spot a stale dump.

    Printed at four decimals: enough to make the 10-decimal dummy placeholders
    obvious at a glance, since a real rollout count over N trials lands on a
    round fraction.

    The figure deliberately shows no numbers; this is where they live."""
    head = f"{'demos':<8}" + "".join(f"{label_for(m):>18}" for m in methods)
    print(f"\n  {task_label}  (success rate, %)")
    print("  " + head)
    print("  " + "-" * len(head))
    for i, n in enumerate(n_demos):
        cells = "".join(
            f"{values[m][i]:>18.4f}" if not np.isnan(values[m][i]) else f"{'--':>18}"
            for m in methods)
        print("  " + f"{int(n):<8}" + cells)


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

    # The style is a swappable module exposing the same names (PALETTE, MARKERS,
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

    task, n_demos, values, methods, is_dummy = load(args.csv)
    task_label = tasks.label(task)
    print_table(task_label, n_demos, values, methods)

    style.apply_style()
    import matplotlib.pyplot as plt

    # Explicit margins, not constrained_layout, because savefig.bbox is off (see
    # style.py): the saved page has to be exactly figsize or the include rescales
    # the text. Left margin fits a two-digit tick plus the y label; bottom fits
    # the tick row plus the x label.
    #
    # The margins are given in inches and converted, not written as fractions
    # directly: subplots_adjust takes fractions of the figure, so a hardcoded
    # fraction silently rescales the text margins whenever the height changes
    # and the x label starts colliding with the page edge.
    h = style.figsize("col", ratio=0.595)[1]
    fig, ax = plt.subplots(figsize=(style.COL_WIDTH, h))
    fig.subplots_adjust(left=0.145, right=0.985,
                        bottom=0.43 / h, top=1 - 0.06 / h)

    for mi, m in enumerate(methods):
        ours = is_ours(m)
        ax.plot(
            n_demos, values[m],
            linestyle=style.LINE_OURS if ours else style.LINE_OTHER,
            linewidth=1.2,
            marker=style.MARKERS[mi % len(style.MARKERS)],
            markersize=4.5,
            color=style.PALETTE[mi % len(style.PALETTE)],
            # Near-black contour for definition; red on "ours" to flag it, same
            # width -- the colour alone is the highlight. "ours" draws on top so
            # its red edge is never occluded where the curves cross.
            markeredgecolor=style.MARKER_EDGE_OURS if ours else style.MARKER_EDGE,
            markeredgewidth=0.6,
            label=label_for(m),
            zorder=4 if ours else 3,
        )

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 25))
    # Pad the x range by 4% of the span on each side so the 0-demo and 50-demo
    # marks are not clipped in half by the spines.
    pad = 0.04 * (n_demos[-1] - n_demos[0])
    ax.set_xlim(n_demos[0] - pad, n_demos[-1] + pad)
    # Ticks only at the budgets actually run -- a tick at 30 would invite the
    # reader to read a value off the connecting line, which is interpolation,
    # not data.
    ax.set_xticks(n_demos)
    ax.set_xticklabels([str(int(n)) for n in n_demos])
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    # The task is named on the axis rather than left to the caption alone: this
    # figure reports one task out of four, and a reader who lands on it from the
    # bar figure has no other way to tell which. Taken from the CSV's task
    # column through the shared table, so it cannot name a different task than
    # the data plotted above it.
    ax.set_xlabel(f"# Robot Demonstrations (Task: {task_label})")
    # Plain "%", not the LaTeX-escaped "\%": both style stacks run with
    # text.usetex off (style_science pulls in SciencePlots' 'no-latex'), so an
    # escaped percent would render as a literal backslash. If you ever drop
    # 'no-latex' to get true LaTeX text, this needs escaping.
    ax.set_ylabel("Success Rate [%]")

    # Legend inside the axes, upper left: both curves rise left-to-right, so
    # that corner is the one region guaranteed to stay empty. Outside placement
    # would cost a row of figure height for two short entries.
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.3,
              handlelength=1.9, labelspacing=0.3)

    style.save(fig, name)

    n_missing = sum(int(np.isnan(v).sum()) for v in values.values())
    if n_missing:
        print(f"\n  note: {n_missing} empty cell(s) not yet run; drawn as gaps")
    if is_dummy:
        print("\n  " + "!" * 66)
        print(f"  ! {args.csv.name} is flagged DUMMY DATA -- these are placeholders,")
        print("  ! not measurements. Do not ship this figure in the paper.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
