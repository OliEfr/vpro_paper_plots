"""Latent-action probing quality: per-dimension R^2, one figure per eval suite.

Reads every ``results/probing_<suite>.csv`` and emits
``figures/probing_<suite>.pdf``. Run with no arguments to rebuild everything:

    python plot_probing.py

Input schema (see results/README.md). One row per action dimension plus an
optional aggregate row where ``action_dim == "mean"``:

    action_dim,axis,<method>_r2,<method>_r2,...

Any column ending in ``_r2`` is treated as a method, in file order, so adding
the remaining 2x2 configurations means adding columns -- no change here.
Derived columns such as ``delta`` and ``rel_gain_pct`` are ignored on read and
recomputed, so the figure can never disagree with the raw R^2 values.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Relative gains are only printed where the reference R^2 clears this bar.
# Below it the probe is barely fitting the axis at all, and a small absolute
# move turns into a huge ratio -- e.g. delta_ry .236 -> .322 reads as "+36.7%"
# while delta_y .872 -> .880 reads as "+1.0%", which invites exactly the wrong
# conclusion about where the method helps. Absolute R^2 is always shown.
REL_GAIN_MIN_R2 = 0.30

# Display names. Keys are the CSV column names; unknown columns fall back to
# the column name with the _r2 suffix stripped.
METHOD_LABELS = {
    "single_r2": "SingleView",
    "multi_r2": "MultiView",
    # Once the full 2x2 lands, dump these columns instead and label them here:
    # "sv_sf_r2": r"$1\times\{0,5\}$",
    # "sv_mf_r2": r"$1\times\{0,1,5,9\}$",
    # "mv_sf_r2": r"$2\times\{0,5\}$",
    # "mv_mf_r2": r"$2\times\{0,1,5,9\}$",
}

AXIS_LABELS = {
    "delta_x": r"$\Delta x$",
    "delta_y": r"$\Delta y$",
    "delta_z": r"$\Delta z$",
    "delta_rx": r"$\Delta r_x$",
    "delta_ry": r"$\Delta r_y$",
    "delta_rz": r"$\Delta r_z$",
    "gripper": "Grip.",
    "all_dims": "Mean",
}

SUITE_LABELS = {
    "libero": "LIBERO",
    "libero_plus": "LIBERO-PLUS",
    "mimicgen": "MimicGen",
}


def load(csv_path):
    """Return (per-dimension frame, mean row or None, method column names)."""
    df = pd.read_csv(csv_path)
    methods = [c for c in df.columns if c.endswith("_r2")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_r2'")

    is_mean = df["action_dim"].astype(str).str.lower() == "mean"
    dims, mean_rows = df[~is_mean].copy(), df[is_mean]

    if mean_rows.empty:
        mean_row = None
    else:
        mean_row = mean_rows.iloc[0]
        # The dumped mean should be the mean over dimensions. If it is not, the
        # dump is stale or was computed over a different dimension set -- worth
        # knowing before it reaches a figure.
        for m in methods:
            recomputed = dims[m].mean()
            if not np.isclose(mean_row[m], recomputed, atol=5e-3):
                print(f"  ! {csv_path.name}: {m} mean row is {mean_row[m]:.4f}, "
                      f"mean over dims is {recomputed:.4f}")
    return dims, mean_row, methods


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_r2") else method)


def suite_from(csv_path):
    stem = re.sub(r"^probing_", "", csv_path.stem)
    return SUITE_LABELS.get(stem, stem.replace("_", "-").upper()), stem


def print_table(suite_label, dims, mean_row, methods):
    """Text table view -- identity is never carried by color alone."""
    ref = methods[0]
    head = f"{'axis':<10}" + "".join(f"{label_for(m):>13}" for m in methods)
    if len(methods) > 1:
        head += f"{'delta':>9}{'rel [%]':>9}"
    print(f"\n  {suite_label}")
    print("  " + head)
    print("  " + "-" * len(head))

    rows = list(dims.iterrows())
    for _, r in rows:
        # Raw axis names here, not AXIS_LABELS -- those carry mathtext markup
        # that is meant for the figure and would wreck the column alignment.
        line = f"{r['axis']:<10}"
        line += "".join(f"{r[m]:>13.4f}" for m in methods)
        if len(methods) > 1:
            d = r[methods[-1]] - r[ref]
            rel = f"{100 * d / r[ref]:+.1f}" if r[ref] > REL_GAIN_MIN_R2 else "--"
            line += f"{d:>+9.4f}{rel:>9}"
        print("  " + line)

    if mean_row is not None:
        line = f"{'Mean':<10}" + "".join(f"{mean_row[m]:>13.4f}" for m in methods)
        if len(methods) > 1:
            d = mean_row[methods[-1]] - mean_row[ref]
            # Reported on the means, not as a mean of the per-dimension ratios.
            # The two differ (a mean of ratios is dominated by the smallest
            # denominator) and the caption must say which one this is.
            line += f"{d:>+9.4f}{100 * d / mean_row[ref]:>+9.1f}"
        print("  " + "-" * len(head))
        print("  " + line)


def plot(csv_path, width=None):
    suite_label, suite_key = suite_from(csv_path)
    dims, mean_row, methods = load(csv_path)
    print_table(suite_label, dims, mean_row, methods)

    import matplotlib.pyplot as plt

    labels = [AXIS_LABELS.get(a, a) for a in dims["axis"]]
    values = [dims[m].to_numpy(dtype=float) for m in methods]
    if mean_row is not None:
        labels.append("Mean")
        values = [np.append(v, float(mean_row[m])) for v, m in zip(values, methods)]

    n_groups, n_methods = len(labels), len(methods)
    # Four methods x eight groups does not fit a single column legibly.
    if width is None:
        width = "col" if n_methods <= 2 else "text"

    # The mean is an aggregate of the bars to its left, not a peer of them, so
    # it gets a gap and a divider rather than sitting flush in the sequence.
    has_mean = mean_row is not None
    x = np.arange(n_groups, dtype=float)
    if has_mean:
        x[-1] += 0.55

    bar_w = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=style.figsize(width, ratio=0.55 if width == "col" else 0.34))

    for i, (m, v) in enumerate(zip(methods, values)):
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(
            x + offset, v, bar_w * 0.88,
            label=label_for(m),
            color=style.PALETTE[i % len(style.PALETTE)],
            hatch=style.HATCHES[i % len(style.HATCHES)],
            # A surface-colored edge keeps adjacent bars from fusing into one
            # block, and keeps the hatch from bleeding across the boundary.
            edgecolor="white", linewidth=0.5,
        )
        # Direct value labels: the palette's lighter steps sit under 3:1 against
        # white, so the numbers -- not the fill -- carry the reading.
        for xi, vi in zip(x + offset, v):
            ax.text(xi, vi + 0.015, f"{vi:.2f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=5.5, color=style.INK_MUTED, rotation=90)

    if has_mean:
        ax.axvline(x[-1] - 0.5, color=style.GRID, linewidth=0.6, zorder=0)

    # Relative gain of the last method over the first, printed only where the
    # reference clears the floor. This is the "how much better" number; the
    # bars themselves carry the absolute values.
    if n_methods > 1:
        ref, last = values[0], values[-1]
        for xi, r, l in zip(x, ref, last):
            if r <= REL_GAIN_MIN_R2:
                continue
            ax.text(xi, max(r, l) + 0.135, f"{100 * (l - r) / r:+.0f}%",
                    ha="center", va="bottom", fontsize=6, color=style.INK)

    ax.set_ylabel(r"probe $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # R^2 tops out at 1.0, but the headroom above it is not wasted -- the value
    # and relative-gain labels live there. Ticks stop at 1.0 so the axis still
    # reads as the true range.
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    # Title left, legend right, same band -- neither one costs a row.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=min(n_methods, 4))
    ax.set_title(suite_label, loc="left", color=style.INK_MUTED, pad=10)

    style.save(fig, f"probing_{suite_key}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="*", type=Path,
                   help="specific result CSVs (default: all results/probing_*.csv)")
    p.add_argument("--width", choices=["col", "text"], default=None,
                   help="figure width: \\columnwidth or \\textwidth (default: auto)")
    args = p.parse_args()

    paths = args.csv or sorted(RESULTS_DIR.glob("probing_*.csv"))
    if not paths:
        raise SystemExit(f"no probing CSVs found in {RESULTS_DIR}")

    style.apply_style()
    for path in paths:
        plot(path, width=args.width)


if __name__ == "__main__":
    main()
