r"""Multi-view versus side-only real-world success at a five-demo budget.

Reads ``results/realworld_multiview_vs_side.csv`` and emits
``figures/realworld_multiview_vs_side.pdf``. Run with no arguments to rebuild:

    python plot_realworld_multiview_vs_side.py

Encoding: y identifies the task, x is success rate in percent, and color plus
hatch identify the view configuration. The horizontal layout leaves room for
the full task names within one IEEE column. The pooled Overall row is separated
from the four task rows by a rule.

EMBEDDING IN LATEX
------------------
Built at 3.487in = \columnwidth (252pt, per IEEEtran journal mode), so it is a
plain figure, not a figure*. Needs \usepackage{graphicx}.

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/realworld_multiview_vs_side.pdf}
      \caption{Real-world success rate at a five-robot-demo budget for frozen
      video representations using multi-view or side-only video. Overall pools
      100 rollouts across four tasks.}
      \label{fig:realworld_multiview_vs_side}
    \end{figure}

Do not rescale it -- see README.md.

Input schema (see results/README.md). One row per task plus an Overall row:

    task,multiview_sr,side_only_sr,n_rollouts

Success rates are stored as fractions in [0, 1]. ``n_rollouts`` is used to
cross-check that Overall is the rollout-weighted pooled result.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "realworld_multiview_vs_side.csv"

METHODS = ["multiview_sr", "side_only_sr"]
METHOD_LABELS = {
    "multiview_sr": "Multi-View",
    "side_only_sr": "Side-Only",
}
TASK_LABELS = {
    "task3": "Banana in\ncardboard box",
    "task4": "Banana in\nblack bowl",
    "task1": "Milk on\npink plate",
    "task2": "Salt on\npink plate",
    "overall": r"$\mathbf{Overall}$",
}


def load(csv_path):
    """Load values and verify the pooled Overall row."""
    df = pd.read_csv(csv_path)
    required = {"task", "n_rollouts", *METHODS}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"{csv_path.name}: missing column(s): {', '.join(sorted(missing))}")

    if df["task"].duplicated().any():
        duplicated = ", ".join(df.loc[df["task"].duplicated(), "task"].astype(str))
        raise SystemExit(f"{csv_path.name}: duplicate task row(s): {duplicated}")

    overall_rows = df[df["task"] == "overall"]
    task_rows = df[df["task"] != "overall"]
    if len(overall_rows) != 1:
        raise SystemExit(f"{csv_path.name}: expected exactly one overall row")
    if task_rows.empty:
        raise SystemExit(f"{csv_path.name}: no task rows")

    for method in METHODS:
        values = df[method].to_numpy(dtype=float)
        if np.isnan(values).any() or np.min(values) < 0 or np.max(values) > 1:
            raise SystemExit(f"{csv_path.name}: {method} must contain fractions in [0, 1]")

        weights = task_rows["n_rollouts"].to_numpy(dtype=float)
        pooled = np.average(task_rows[method].to_numpy(dtype=float), weights=weights)
        dumped = float(overall_rows.iloc[0][method])
        if not np.isclose(pooled, dumped, atol=1e-9):
            raise SystemExit(
                f"{csv_path.name}: {method} overall is {dumped:.4f}, "
                f"but pooling the task rows gives {pooled:.4f}"
            )

    expected_rollouts = int(task_rows["n_rollouts"].sum())
    dumped_rollouts = int(overall_rows.iloc[0]["n_rollouts"])
    if dumped_rollouts != expected_rollouts:
        raise SystemExit(
            f"{csv_path.name}: overall n_rollouts is {dumped_rollouts}, "
            f"but task rows sum to {expected_rollouts}"
        )

    return df


def print_table(df):
    head = f"{'task':<25}" + "".join(f"{METHOD_LABELS[m]:>14}" for m in METHODS)
    print("\n  Five-demo frozen-video success rate, %")
    print("  " + head)
    print("  " + "-" * len(head))
    for row in df.itertuples(index=False):
        label = TASK_LABELS.get(row.task, str(row.task)).replace("\n", " ")
        label = label.replace(r"$\mathbf{", "").replace("}$", "")
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
    # five compact groups. Margins are explicit because savefig.bbox is off.
    h = 2.45
    fig, ax = plt.subplots(figsize=(style.COL_WIDTH, h))
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.16, top=0.84)

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

    # Overall is a pooled summary rather than a fifth task.
    overall_index = int(df.index[df["task"] == "overall"][0])
    ax.axhline(overall_index - 0.5, color=style.INK_MUTED, linewidth=0.8, zorder=2)

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
        bbox_to_anchor=(0.34, 0.99),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.4,
    )

    style.save(fig, name)


if __name__ == "__main__":
    main()
