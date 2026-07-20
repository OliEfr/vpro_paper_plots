"""Latent-action probing quality: per-dimension R^2 across LAM configurations.

Reads every ``results/probing_<suite>.csv`` and emits ``figures/probing.pdf``,
one stacked panel per eval suite sharing a single legend and x axis. Run with
no arguments to rebuild:

    python plot_probing.py

Input schema (see results/README.md). One row per action dimension plus an
optional aggregate row where ``action_dim == "mean"``:

    action_dim,axis,<method>_r2,<method>_r2,...

Any column ending in ``_r2`` is treated as a method, in file order, so adding
or dropping a LAM configuration is a schema change only. Values of exactly 0
are read as "not run yet" rather than as a measurement -- see PLACEHOLDER.

Only configurations that *have* a LAM appear here. The action-only policy has
no latent action model, so there is no latent space to probe; its row belongs
in the policy-performance table, not in this figure.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Experiments that have not been run yet are dumped as 0. A probe R^2 of
# exactly 0.0 is not a plausible measurement (it would mean the probe exactly
# matches predicting the mean), so treating 0 as a placeholder is safe -- and
# the alternative, drawing it as a real point on the baseline, would read as
# "this configuration scores zero" rather than "we have no number yet".
PLACEHOLDER = 0.0

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _r2 suffix stripped. Order here does not matter; the CSV column
# order is what fixes the plotting order and the palette assignment.
METHOD_LABELS = {
    "sv_sf_r2": r"$1\times\{0,5\}$",
    "sv_mf_r2": r"$1\times\{0,1,5,9\}$",
    "mv_sf_r2": r"$2\times\{0,5\}$",
    "mv_mf_r2": r"$2\times\{0,1,5,9\}$ (ours)",
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

# Panels are drawn in this order when present; anything else follows, sorted.
SUITE_ORDER = ["libero", "libero_plus", "mimicgen"]


def suite_from(csv_path):
    stem = re.sub(r"^probing_", "", csv_path.stem)
    return SUITE_LABELS.get(stem, stem.replace("_", "-").upper()), stem


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_r2") else method)


def load(csv_path):
    """Return (labels, {method: values}, methods, n_placeholder).

    Values are NaN wherever the dump carried the placeholder, so downstream
    code never has to special-case it -- matplotlib skips NaN, and nanmean
    ignores it.
    """
    df = pd.read_csv(csv_path)
    methods = [c for c in df.columns if c.endswith("_r2")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_r2'")

    is_mean = df["action_dim"].astype(str).str.lower() == "mean"
    dims, mean_rows = df[~is_mean], df[is_mean]

    labels = list(dims["axis"])
    values = {m: dims[m].to_numpy(dtype=float) for m in methods}

    if not mean_rows.empty:
        labels.append("all_dims")
        mean_row = mean_rows.iloc[0]
        for m in methods:
            # Check the dumped mean against the mean over dimensions before
            # trusting it. A mismatch usually means a stale dump.
            dumped, recomputed = float(mean_row[m]), np.nanmean(
                np.where(values[m] == PLACEHOLDER, np.nan, values[m]))
            if (dumped != PLACEHOLDER and not np.isnan(recomputed)
                    and not np.isclose(dumped, recomputed, atol=5e-3)):
                print(f"  ! {csv_path.name}: {m} mean row is {dumped:.4f}, "
                      f"mean over dims is {recomputed:.4f}")
            values[m] = np.append(values[m], dumped)

    n_placeholder = sum(int((v == PLACEHOLDER).sum()) for v in values.values())
    values = {m: np.where(v == PLACEHOLDER, np.nan, v) for m, v in values.items()}
    return labels, values, methods, n_placeholder


def print_table(suite_label, labels, values, methods):
    """Text table view -- identity is never carried by color alone, and this
    is also the fastest way to spot a stale dump.

    The figure deliberately shows no numbers; this is where they live."""
    # Plain column stems, not METHOD_LABELS -- those carry mathtext markup
    # meant for the figure, which would wreck the column alignment here.
    head = f"{'axis':<10}" + "".join(f"{m[:-3]:>10}" for m in methods)
    print(f"\n  {suite_label}")
    print("  " + head)
    print("  " + "-" * len(head))
    for i, axis in enumerate(labels):
        if axis == "all_dims":
            print("  " + "-" * len(head))
        cells = "".join(
            f"{values[m][i]:>10.4f}" if not np.isnan(values[m][i]) else f"{'--':>10}"
            for m in methods)
        print("  " + f"{axis:<10}" + cells)


def draw_panel(ax, suite_label, labels, values, methods, show_xticks):
    n_groups, n_methods = len(labels), len(methods)

    # The mean is an aggregate of the groups to its left, not a peer of them,
    # so it gets a gap and a divider rather than sitting flush in the sequence.
    has_mean = labels and labels[-1] == "all_dims"
    x = np.arange(n_groups, dtype=float)
    if has_mean:
        x[-1] += 0.6

    # Spread the four marks across the group without letting them touch.
    span = 0.62
    offsets = np.linspace(-span / 2, span / 2, n_methods) if n_methods > 1 else np.zeros(1)

    for i, m in enumerate(methods):
        ax.plot(
            x + offsets[i], values[m],
            linestyle="none",
            marker=style.MARKERS[i % len(style.MARKERS)],
            markersize=4.2,
            color=style.PALETTE[i % len(style.PALETTE)],
            # Surface-colored ring keeps overlapping marks from fusing.
            markeredgecolor="white", markeredgewidth=0.6,
            label=label_for(m), zorder=3,
        )

    if has_mean:
        ax.axvline(x[-1] - 0.5, color=style.GRID, linewidth=0.6, zorder=0)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in labels] if show_xticks else [])
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"probe $R^2$")
    # Suite name inside the panel: with three stacked panels a title per panel
    # would cost three rows of vertical space for three words.
    ax.text(0.995, 0.93, suite_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=style.INK)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", nargs="*", type=Path,
                   help="specific result CSVs (default: all results/probing_*.csv)")
    p.add_argument("--name", default="probing", help="output basename in figures/")
    args = p.parse_args()

    paths = args.csv or sorted(RESULTS_DIR.glob("probing_*.csv"))
    if not paths:
        raise SystemExit(f"no probing CSVs found in {RESULTS_DIR}")
    order = {s: i for i, s in enumerate(SUITE_ORDER)}
    paths = sorted(paths, key=lambda q: (order.get(suite_from(q)[1], len(order)), q.stem))

    style.apply_style()
    import matplotlib.pyplot as plt

    panels = []
    total_placeholder = 0
    for path in paths:
        suite_label, _ = suite_from(path)
        labels, values, methods, n_ph = load(path)
        total_placeholder += n_ph
        print_table(suite_label, labels, values, methods)
        panels.append((suite_label, labels, values, methods))

    # Full text width: four marks across eight groups needs the room, and the
    # figure carries all three suites, so it belongs in a figure* anyway.
    n = len(panels)
    fig, axes = plt.subplots(
        n, 1, sharex=True,
        figsize=(style.TEXT_WIDTH, 1.05 * n + 0.55),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for i, (ax, (suite_label, labels, values, methods)) in enumerate(zip(axes, panels)):
        draw_panel(ax, suite_label, labels, values, methods, show_xticks=(i == n - 1))

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="outside upper center",
               ncol=len(lbls), frameon=False)

    style.save(fig, args.name)
    if total_placeholder:
        print(f"\n  note: {total_placeholder} placeholder cell(s) not yet run; "
              f"drawn as gaps, not as zeros")


if __name__ == "__main__":
    main()
