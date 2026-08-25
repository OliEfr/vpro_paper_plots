r"""Multi-view versus side-only real-world success at a five-demo budget.

Reads ``results/realworld_multiview_vs_side.csv`` and emits
``figures/realworld_multiview_vs_side.pdf``. Run with no arguments to rebuild:

    python plot_realworld_multiview_vs_side.py

Encoding: y identifies the task, x is success rate in percent, and color plus
hatch identify the view configuration. The horizontal layout leaves room for
the full task names within one IEEE column.

The figure carried a pooled Overall row until 2026-08-25 and no longer does --
it is per-task only. Pooling four tasks of 25 rollouts each hid that the gap is
not uniform across them, and the row read as a fifth task however it was ruled
off. If a summary number is wanted, put it in the caption where it cannot be
mistaken for a measurement of its own.

EMBEDDING IN LATEX
------------------
Built at 3.487in = \columnwidth (252pt, per IEEEtran journal mode), so it is a
plain figure, not a figure*. Needs \usepackage{graphicx}.

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/realworld_multiview_vs_side.pdf}
      \caption{Real-world success rate at a five-robot-demo budget for frozen
      video representations, using multi-view or single-view video. 25 rollouts
      per task.}
      \label{fig:realworld_multiview_vs_side}
    \end{figure}

Do not rescale it -- see README.md.

Input schema (see results/README.md). One row per task:

    task,multiview_sr,side_only_sr,n_rollouts

Success rates are stored as fractions in [0, 1]. ``n_rollouts`` is carried per
task; it is the denominator behind each rate rather than something this figure
draws.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "realworld_multiview_vs_side.csv"

METHODS = ["multiview_sr", "side_only_sr"]
# The column stays `side_only_sr` while the label reads "Single-view" -- the CSV
# keys on the measurement, the figure shows the name the paper uses, and this
# mapping is where the two are allowed to differ.
METHOD_LABELS = {
    "multiview_sr": "Multi-View",
    "side_only_sr": "Single-view",
}
TASK_LABELS = {
    "task3": "Banana in\ncardboard box",
    "task4": "Banana in\nblack bowl",
    "task1": "Milk on\npink plate",
    "task2": "Salt on\npink plate",
}


def load(csv_path):
    """Load and validate the per-task rows.

    An `overall` row used to be required here and cross-checked against the
    rollout-weighted pooling of the task rows. The figure no longer draws one,
    so the row is now rejected rather than ignored: silently dropping it would
    let a dump carry a pooled number that nothing checks any more.
    """
    df = pd.read_csv(csv_path)
    required = {"task", "n_rollouts", *METHODS}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{csv_path.name}: missing column(s): {', '.join(sorted(missing))}")

    if df["task"].duplicated().any():
        duplicated = ", ".join(df.loc[df["task"].duplicated(), "task"].astype(str))
        raise SystemExit(f"{csv_path.name}: duplicate task row(s): {duplicated}")

    if (df["task"] == "overall").any():
        raise SystemExit(
            f"{csv_path.name}: holds an 'overall' row, which this figure no "
            f"longer draws -- delete it, and put the pooled number in the "
            f"caption if the paper needs one")
    if df.empty:
        raise SystemExit(f"{csv_path.name}: no task rows")

    for method in METHODS:
        values = df[method].to_numpy(dtype=float)
        if np.isnan(values).any() or np.min(values) < 0 or np.max(values) > 1:
            raise SystemExit(f"{csv_path.name}: {method} must contain fractions in [0, 1]")

    return df


def print_table(df):
    head = f"{'task':<25}" + "".join(f"{METHOD_LABELS[m]:>14}" for m in METHODS)
    print("\n  Five-demo frozen-video success rate, %")
    print("  " + head)
    print("  " + "-" * len(head))
    for row in df.itertuples(index=False):
        label = TASK_LABELS.get(row.task, str(row.task)).replace("\n", " ")
        cells = "".join(f"{getattr(row, m) * 100:>14.1f}" for m in METHODS)
        print("  " + f"{label:<25}" + cells)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--name", default=None)
    parser.add_argument("--style", choices=["paper", "science"], default="paper")
    args = parser.parse_args()

    global style
    name = args.name or args.csv.stem
    if args.style == "science":
        import style_science

        style = style_science
        if args.name is None:
            name += "_science"

    if not args.csv.exists():
        raise SystemExit(f"no such results file: {args.csv}")

    df = load(args.csv)
    print_table(df)

    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_rows = len(df)
    y = np.arange(n_rows, dtype=float)
    bar_h = 0.32

    # The horizontal layout spends width on readable task names and height on
    # the task rows. Margins are explicit because savefig.bbox is off.
    #
    # Height follows the row count at a fixed pitch, rather than the rows being
    # divided into a constant height: dropping the Overall row would otherwise
    # have stretched the remaining four to fill the same 2.45in, which changes
    # the bar density of a figure whose data did not change.
    # top_in is the legend band: one row of text plus its gap to the axes, and
    # no more. It was 0.39in when the figure was 2.45in tall and the slack was
    # not obvious; at four rows the same band reads as a hole above the plot.
    row_in, top_in, bottom_in = 0.333, 0.27, 0.39
    h = n_rows * row_in + top_in + bottom_in
    # right_in holds the half-width of the last x tick label. At the old 0.02
    # fraction (0.07in) the "100" overran the canvas by 3px and printed shaved,
    # which is the clipping the no-bbox_inches rule in README.md warns about.
    right_in = 0.11
    fig, ax = plt.subplots(figsize=(style.COL_WIDTH, h))
    fig.subplots_adjust(left=0.34, right=1 - right_in / style.COL_WIDTH,
                        bottom=bottom_in / h, top=1 - top_in / h)

    # Keep the established paper semantics: the stronger/ours arm is green and
    # hatched, the baseline is blue and plain. Hue never carries identity alone.
    method_style = {
        "multiview_sr": (style.PALETTE[1], style.HATCHES[1]),
        "side_only_sr": (style.PALETTE[0], style.HATCHES[0]),
    }
    offsets = {"multiview_sr": -bar_h / 2, "side_only_sr": bar_h / 2}
    for method in METHODS:
        color, hatch = method_style[method]
        ax.barh(
            y + offsets[method],
            df[method].to_numpy(dtype=float) * 100,
            height=bar_h,
            facecolor=color,
            edgecolor=style.MARKER_EDGE,
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
        )

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 25))
    ax.set_yticks(y)
    ax.set_yticklabels([TASK_LABELS.get(str(task), str(task)) for task in df["task"]])
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Success Rate [%]")

    handles = [
        Patch(
            facecolor=method_style[m][0],
            edgecolor=style.MARKER_EDGE,
            linewidth=0.6,
            hatch=method_style[m][1],
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.34, 1.0),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.4,
    )

    style.save(fig, name)


if __name__ == "__main__":
    main()
