r"""Latent-action probing quality: per-dimension R^2, all suites and all LAM
configurations in one axes.

Reads every ``results/probing_<suite>.csv`` and emits ``figures/probing.pdf``.
Run with no arguments to rebuild:

    python plot_probing.py

Encoding: x groups by action dimension, **color** identifies the evaluation
suite, **marker shape** identifies the LAM configuration. Two channels for two
independent factors, so a reader can hold one fixed and scan the other.

EMBEDDING IN LATEX
------------------
Built at 7.14in = \textwidth (43pc, per IEEEtran journal mode), so it spans both columns and needs figure*,
not figure. Needs \usepackage{graphicx}.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/probing.pdf}
      \caption{Latent action probing quality. A frozen-latent MLP probe
      reconstructs ground-truth robot actions; we report per-dimension $R^2$
      (higher is better). Color identifies the evaluation suite, marker shape
      the LAM input $V\times\Delta$ (viewpoints $\times$ frame offsets).
      \textbf{Mean} aggregates over all seven action dimensions.}
      \label{fig:probing}
    \end{figure*}

Do not rescale it. Every label is set at the document's 10pt body size in
Computer Modern, the same typeface main.tex renders in, so at width=\textwidth
figure text matches the surrounding prose exactly. Any width= other than
\textwidth scales that text off body size.

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
# order is what fixes the plotting order and the marker assignment.
#
# Brackets, not the set braces the paper uses for Delta. All figure text is one
# size (10pt), but Computer Modern's math brace \{ \} is a tall glyph built to
# wrap fractions: measured, "$1\times\{0,5\}$" renders 13.3pt against 9.5pt for
# every other label, so the legend reads as a larger font even though it is not.
# Brackets render at 11.8pt and sit much closer to the tick-label height.
# (Text-mode braces do not work here: matplotlib's raw cmr10 has no glyph at the
# ASCII { } slots, so "$1\times${0,5}" renders as garbage.)
# To match the paper's set notation exactly at the cost of the taller row, swap
# [ and ] back for \{ and \} inside the math.
METHOD_LABELS = {
    "sv_sf_r2": r"$1\times[0,5]$",
    "sv_mf_r2": r"$1\times[0,1,5,9]$",
    "mv_sf_r2": r"$2\times[0,5]$",
    "mv_mf_r2": r"$2\times[0,1,5,9]$ (ours)",
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

# Suites are drawn in this order when present; anything else follows, sorted.
# This also fixes the color assignment, so the order is not cosmetic -- keep it
# stable across figures or the same suite changes color between them.
SUITE_ORDER = ["libero", "libero_plus", "mimicgen"]


def suite_from(csv_path):
    stem = re.sub(r"^probing_", "", csv_path.stem)
    return SUITE_LABELS.get(stem, stem.replace("_", "-").upper()), stem


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_r2") else method)


def load(csv_path):
    """Return (axis labels, {method: values}, methods, n_placeholder).

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


def build_legends(fig, suite_labels, methods, left):
    """Two legends, one per encoding channel, stacked above the axes.

    A single combined legend would need suites x methods entries to say what
    two short lists say directly, and it would imply the two factors are one.

    Placement is explicit rather than via constrained_layout: two legends both
    asking for "outside upper center" are given the same slot and silently
    drawn on top of each other.
    """
    from matplotlib.lines import Line2D

    common = dict(linestyle="none", markeredgecolor="white", markeredgewidth=0.6,
                  markersize=6)
    suite_handles = [
        Line2D([], [], marker="o",
               color=style.PALETTE[i % len(style.PALETTE)], label=s, **common)
        for i, s in enumerate(suite_labels)
    ]
    # Neutral gray for the shape legend: these entries are about the marker
    # shape only, and coloring them would suggest a suite pairing that is not
    # there.
    method_handles = [
        Line2D([], [], marker=style.MARKERS[i % len(style.MARKERS)],
               color=style.INK_MUTED, label=label_for(m), **common)
        for i, m in enumerate(methods)
    ]

    # Neither legend carries a title. A title on one and not the other left a
    # ragged edge -- the untitled row starts with a marker at the margin while
    # the titled row starts with text and indents its markers past it. Titling
    # both costs two more rows of height. The caption already states that color
    # is the suite and shape is the LAM input, so the titles were repeating it.
    for handles, y in ((suite_handles, 1.0), (method_handles, 0.925)):
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, y),
                   ncol=len(handles), frameon=False, borderaxespad=0)


def main():
    # Raw formatter: the default one reflows paragraphs, which would collapse
    # the LaTeX snippet above into an unusable single block.
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    suites, total_placeholder = [], 0
    for path in paths:
        suite_label, _ = suite_from(path)
        labels, values, methods, n_ph = load(path)
        total_placeholder += n_ph
        print_table(suite_label, labels, values, methods)
        suites.append((suite_label, labels, values, methods))

    # Every suite must agree on the axis rows and the method columns, or the
    # shared x groups and the shared marker legend would both be lying.
    ref_labels, ref_methods = suites[0][1], suites[0][3]
    for suite_label, labels, _, methods in suites[1:]:
        if labels != ref_labels:
            raise SystemExit(f"{suite_label}: axis rows differ from {suites[0][0]}")
        if methods != ref_methods:
            raise SystemExit(f"{suite_label}: method columns differ from {suites[0][0]}")

    n_groups, n_suites, n_methods = len(ref_labels), len(suites), len(ref_methods)

    # The mean is an aggregate of the groups to its left, not a peer of them,
    # so it gets a gap and a divider rather than sitting flush in the sequence.
    has_mean = ref_labels[-1] == "all_dims"
    x = np.arange(n_groups, dtype=float)
    if has_mean:
        x[-1] += 0.65

    # Suite-major ordering: each suite's methods stay contiguous, so a group
    # reads as n_suites colored clusters rather than n_methods interleaved
    # ones. Scanning "how does this suite respond to the configurations" is the
    # comparison the figure is for; scanning across suites is secondary.
    n_slots = n_suites * n_methods
    span = 0.80
    slots = np.linspace(-span / 2, span / 2, n_slots)

    # Explicit margins, not constrained_layout: the two stacked legends are
    # placed by hand (see build_legends), so the space they need has to be
    # reserved by hand too. The two legend rows above the axes take a fixed
    # ~0.45in and the x tick labels below take ~0.25in; the rest is plot area.
    # Sized so the plot area is ~1.79in tall.
    left, bottom, top = 0.092, 0.101, 0.820
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 2.485))
    fig.subplots_adjust(left=left, right=0.995, bottom=bottom, top=top)

    # Alternating group bands. With twelve marks per group the eye needs the
    # group boundary drawn rather than inferred from spacing alone.
    for i in range(0, n_groups, 2):
        ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=style.GRID, alpha=0.25,
                   linewidth=0, zorder=0)

    for si, (_, _, values, methods) in enumerate(suites):
        for mi, m in enumerate(methods):
            ax.plot(
                x + slots[si * n_methods + mi], values[m],
                linestyle="none",
                marker=style.MARKERS[mi % len(style.MARKERS)],
                markersize=5.0,
                color=style.PALETTE[si % len(style.PALETTE)],
                # Surface-colored ring keeps overlapping marks separable.
                markeredgecolor="white", markeredgewidth=0.5,
                zorder=3,
            )

    if has_mean:
        ax.axvline(x[-1] - 0.575, color=style.INK_MUTED, linewidth=0.6, zorder=1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABELS.get(a, a) for a in ref_labels])
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"probe $R^2$")

    build_legends(fig, [s[0] for s in suites], ref_methods, left)
    style.save(fig, args.name)
    if total_placeholder:
        print(f"\n  note: {total_placeholder} placeholder cell(s) not yet run; "
              f"drawn as gaps, not as zeros")


if __name__ == "__main__":
    main()
