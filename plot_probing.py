r"""Latent-action probing quality: per-dimension R^2, all suites and all LAM
configurations in one axes.

Reads every ``results/probing_<suite>.csv`` and emits ``figures/probing.pdf``.
Run with no arguments to rebuild:

    python plot_probing.py

Encoding: x groups by action dimension, a **tinted background band** identifies
the evaluation suite, and **marker shape** identifies the LAM configuration. Two
channels for two independent factors, so a reader can hold one fixed and scan
the other.

WHY THE SUITE IS A BAND AND NOT A COLOURED MARK. Every x group holds n_suites x
n_methods marks -- twelve at present -- and twelve marks cannot be given more
than a twelfth of a group, whatever else is tuned. That pitch is around 4.4pt
here, so the marks have to be ~3.6pt, and a 3.6pt mark is too small to carry
hue and shape at once: the reader is being asked to resolve twelve colour-shape
combinations at a size where the colour is a few pixels of fill.

Moving the suite onto a background band fixes the cause rather than the
symptom. Colour on an area is legible at a fraction of the saturation a mark
needs, so the band can be tinted lightly enough to sit under the data, and the
marks come back as one ink with shape as their only channel -- four shapes to
tell apart instead of twelve combinations. It also frees the marker edge, which
is what let the size come down without the shapes turning into blobs.

The palette rule still holds, because hue is not carrying the suite alone: the
bands sit in the same left-to-right order in every group, and that order reads
in a grayscale photocopy that flattens three light tints into one.

Not connected: the four configurations within a band are four separate training
runs, not a series. A line through them would draw a trend between points that
have no continuum to be on.

EMBEDDING IN LATEX
------------------
Built at 7.14in = \textwidth (43pc, per IEEEtran journal mode), so it spans both columns and needs figure*,
not figure. Needs \usepackage{graphicx}.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/probing.pdf}
      \caption{Latent action probing quality. A frozen-latent MLP probe
      reconstructs ground-truth robot actions; we report per-dimension $R^2$
      (higher is better). The shaded band identifies the evaluation suite --
      the three sit in the same order within every dimension -- and marker
      shape the LAM input $V\times\Delta$ (viewpoints $\times$ frame offsets).
      \textbf{Mean} aggregates over all seven action dimensions.}
      \label{fig:probing}
    \end{figure*}

Do not rescale it. Every label is set at 8pt (IEEEtran \footnotesize, the
caption size -- a step below the 10pt body) in Computer Modern, the typeface
main.tex renders in, so at width=\textwidth the labels land on the page at
exactly that size. Any width= other than \textwidth scales the text off it.

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
    "gripper": "Gripper",
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


def is_ours(method):
    """The hero configuration, flagged by "(ours)" in its display label. Kept in
    one place so the red-edge highlight in the plot and the legend never drift."""
    return "(ours)" in label_for(method)


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


# Suite bands. The tint is the palette hue lightened towards white rather than
# drawn at low alpha, so the band is an opaque colour with a known value: alpha
# would let the y grid show through and would composite differently over the
# white page than over the axes face.
#
# 0.16 is as strong as the band goes. Above that it starts competing with the
# marks sitting on it, which is the failure this whole encoding exists to avoid;
# below it the three suites stop being separable on a small band. Colour-on-area
# tolerates far less saturation than colour-on-mark and still reads, which is
# the reason the suite moved off the marks in the first place.
BAND_TINT = 0.16


def band_colour(i):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(style.PALETTE[i % len(style.PALETTE)])
    return tuple(1 - BAND_TINT * (1 - c) for c in (r, g, b))


def mark_colour(method):
    """Marks carry no suite information, so their colour is free to mean one
    thing only: whether this is the hero configuration."""
    return style.MARKER_EDGE_OURS if is_ours(method) else style.MARKER_EDGE


def layout(n_groups, n_suites, n_methods, has_mean, axes_width_in):
    """Mark positions, in group units, plus the resulting pitch in points.

    A group is 1.0 wide and holds n_suites bands side by side, filling it edge
    to edge; each band holds n_methods marks on a regular pitch, padded by half
    a pitch at both ends so a mark never touches a band edge.

    No white gap between groups: horizontal room is the scarce resource here, so
    the bands run the full group width and a dashed rule (drawn in main) marks
    each group boundary instead. A gap would spend page width on separation that
    a hairline gives for free.
    """
    band_w = 1.0 / n_suites
    pitch = band_w / n_methods
    first = -0.5

    x = np.arange(n_groups, dtype=float)
    if has_mean:
        x[-1] += 0.1125
    span_units = (x[-1] + 0.5) - (x[0] - 0.5)
    pitch_pt = pitch * (axes_width_in / span_units) * 72

    bands = [(first + si * band_w, first + (si + 1) * band_w)
             for si in range(n_suites)]
    slots = [[first + si * band_w + (mi + 0.5) * pitch for mi in range(n_methods)]
             for si in range(n_suites)]
    return x, bands, slots, pitch_pt


def build_legends(fig, suite_labels, methods, left):
    """Two legends, one per encoding channel, stacked above the axes.

    A single combined legend would need suites x methods entries to say what
    two short lists say directly, and it would imply the two factors are one.

    Placement is explicit rather than via constrained_layout: two legends both
    asking for "outside upper center" are given the same slot and silently
    drawn on top of each other.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # The suite key is a swatch, not a mark, because in the plot the suite is
    # the tinted band behind the marks. A marker-shaped key would point at the
    # one thing on the figure that does NOT vary by suite.
    suite_handles = [
        Patch(facecolor=band_colour(i), edgecolor="none", label=s)
        for i, s in enumerate(suite_labels)
    ]
    # Marks are ink, not colour -- shape is their only channel, so the key
    # shows them exactly as drawn: near-black, red on "ours".
    method_handles = [
        Line2D([], [], marker=style.MARKERS[i % len(style.MARKERS)],
               color=mark_colour(m), markeredgecolor="none",
               linestyle="none", markersize=5,
               label=label_for(m))
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
    p.add_argument("--style", choices=["paper", "science"], default="paper",
                   help="paper: exact IEEEtran/Computer-Modern match (default). "
                        "science: the SciencePlots aesthetic (garrettj403/SciencePlots).")
    args = p.parse_args()

    # The style is a swappable module exposing the same names (PALETTE, MARKERS,
    # apply_style, save, ...). Rebinding the module-global `style` here means the
    # draw functions, which look it up at call time, pick up whichever was asked
    # for without any per-call plumbing.
    global style
    if args.style == "science":
        import style_science
        style = style_science
        if args.name == "probing":  # keep the two styles' outputs side by side
            args.name = "probing_science"

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

    # Explicit margins, not constrained_layout: the two stacked legends are
    # placed by hand (see build_legends), so the space they need has to be
    # reserved by hand too. The two legend rows above the axes take a fixed
    # ~0.45in and the x tick labels below take ~0.25in; the rest is plot area.
    # Sized so the plot area is ~1.79in tall.
    #
    # The left margin is 0.046 rather than the 0.092 a horizontal tick label
    # needs, because the tick labels are rotated upright below -- that is half
    # an inch of page that goes straight into the group width, where twelve
    # marks are competing for it.
    left, bottom, top = 0.046, 0.101, 0.820
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 2.485))
    fig.subplots_adjust(left=left, right=0.995, bottom=bottom, top=top)

    # The mean is an aggregate of the groups to its left, not a peer of them, so
    # layout() gives it a gap and it gets a divider drawn down the middle of it.
    has_mean = ref_labels[-1] == "all_dims"
    x, bands, slots, pitch_pt = layout(
        n_groups, n_suites, n_methods, has_mean,
        axes_width_in=(0.995 - left) * style.TEXT_WIDTH)

    MARKSIZE = 3.6
    if pitch_pt < MARKSIZE:
        print(f"\n  ! marks are {MARKSIZE}pt on a {pitch_pt:.2f}pt pitch -- they "
              f"overlap. {n_suites} suites x {n_methods} configs no longer fit a "
              f"group; drop a series or the Mean column.")

    # Suite bands: one tinted stripe per (group, suite), full axes height. This
    # is the whole point of the encoding -- colour identifies the suite on an
    # area, where it is easy to see and survives being lightened, so the marks
    # themselves carry no colour and shape is their only channel.
    #
    # Hue is not alone in doing it, which the palette rule requires: the suites
    # sit in the same left-to-right order in every group, so position identifies
    # a band as well as its tint does, and the order survives a grayscale
    # photocopy that flattens three light tints into one.
    for gi in range(n_groups):
        for si, (b0, b1) in enumerate(bands):
            ax.axvspan(x[gi] + b0, x[gi] + b1, color=band_colour(si),
                       linewidth=0, zorder=0)

    for si, (_, _, values, methods) in enumerate(suites):
        for mi, m in enumerate(methods):
            ours = is_ours(m)
            ax.plot(
                x + slots[si][mi], values[m],
                linestyle="none",
                marker=style.MARKERS[mi % len(style.MARKERS)],
                # No edge: at this size a 0.5pt contour is a third of the mark's
                # width, and against a band tinted to 0.16 the fill already has
                # all the contrast it needs. Dropping it is also what lets the
                # shapes stay lean without turning into blobs.
                markersize=MARKSIZE,
                color=mark_colour(m),
                markeredgecolor="none",
                # "ours" on top, so the one mark a reader is looking for is
                # never the one hidden under a neighbour.
                zorder=4 if ours else 3,
            )

    # Group separators. With the bands filling each group there is no white gap
    # left to show where one action dimension ends and the next begins, so a
    # dashed rule draws every dimension-to-dimension boundary. The grip|mean
    # boundary is drawn solid below instead: the mean is an aggregate of the
    # dimensions to its left, not another dimension, and the heavier, continuous
    # rule is what says so.
    n_regular = n_groups - 1 if has_mean else n_groups
    for gi in range(n_regular - 1):
        ax.axvline(x[gi] + 0.5, color=style.INK_MUTED, linewidth=0.6,
                   linestyle=(0, (4, 3)), zorder=1)

    if has_mean:
        ax.axvline(x[-1] - 0.5 - (x[-1] - x[-2] - 1) / 2,
                   color=style.INK_MUTED, linewidth=0.6, zorder=1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    # Upright, not lying flat: rotated, a tick label is one cap-height wide
    # instead of four characters, which is the left margin above.
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], rotation=90,
                       va="center")
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
