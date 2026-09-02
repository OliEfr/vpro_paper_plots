r"""Latent-action probing quality: per-dimension R^2, all suites and all LAM
configurations in one axes.

Reads every ``results/probing_<suite>.csv`` and emits ``figures/probing.pdf``.
Run with no arguments to rebuild:

    python plot_probing.py

TWO PROBES, TWO FIGURES. ``--probe ridge`` draws the linear-probe twin from
``results/probing_ridge/`` into ``figures/probing_ridge.pdf`` -- same suites,
same arms, same rows, same figure, one metric changed. It is a separate figure
and not extra series in this one because a probe and a LAM are not the same kind
of thing: the marks here mean "which latent-action model", and a ridge column
drawn beside them would put "which probe" on the same channel. Kept comparable
on purpose -- identical y range, so the ridge figure visibly sits lower and the
two can be read against each other rather than each rescaled to fill its box.

Encoding: x groups by action dimension, a **tinted background band** identifies
the evaluation suite, and **marker shape** identifies the LAM configuration. Two
channels for two independent factors, so a reader can hold one fixed and scan
the other.

WHY THE SUITE IS A BAND AND NOT A COLOURED MARK. Every x group holds n_suites x
n_methods marks -- twelve at present -- and twelve marks cannot be given more
than a twelfth of a group, whatever else is tuned. That pitch is 5.0pt here, so
the marks come out at 3.6pt, and a 3.6pt mark is too small to carry hue and
shape at once: the reader is being asked to resolve twelve colour-shape
combinations at a size where the colour is a few pixels of fill. (The mark size
is derived from the pitch rather than fixed, so those two numbers move together
when a suite is added or dropped -- see MARKSIZE.)

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
      the three sit in the same order within every dimension -- and marker shape
      the latent-action model being probed. All four share the same two-frame
      input $\Delta=[0,5]$, so the only things that vary are the viewpoint count
      and what trains the latent. \textbf{Mean} aggregates over all seven action
      dimensions. Note that $R^2$ here does \emph{not} track success rate: on
      LIBERO-PLUS the grounded (CLAM-style) teacher probes highest of the four
      and its policy is the weakest of the four.}
      %% LIBERO's band is PLACEHOLDER DATA -- do not submit with it in.
      \label{fig:probing}
    \end{figure*}

The ridge twin embeds with the same snippet, swapping the file and the label
for ``probing_ridge`` and saying *linear* probe in the caption. Its own result
is worth a sentence: on MimicGen the linear probe **reorders the arms**. Under
the MLP probe the LAOF-style flow arm is second of five (0.5577, behind
multi-view's 0.5585); under ridge it is **last** (0.3349, below the single-view
arm it ablates), and the DINOv3 arm likewise falls below single-view. Those are
the two arms whose policies scored below single-view -- so on this suite the
linear probe ranks the arms closer to their success rates than the MLP probe
does. On LIBERO-PLUS the two probes agree on the order. Do not turn that into a
law from two suites; it is a reason to report both, not to pick one.

Do not rescale it. Every label is set at 8pt (IEEEtran \footnotesize, the
caption size -- a step below the 10pt body) in Computer Modern, the typeface
main.tex renders in, so at width=\textwidth the labels land on the page at
exactly that size. Any width= other than \textwidth scales the text off it.

Input schema (see results/README.md). One row per action dimension plus an
optional aggregate row where ``action_dim == "mean"``:

    action_dim,axis,<method>_r2,<method>_r2,...

Any column ending in ``_r2`` is treated as a method, in file order, so adding
a LAM configuration is a schema change only. Values of exactly 0 are read as
"not run yet" rather than as a measurement -- see PLACEHOLDER.

Dropping one from the *figure* is not a schema change: put its column name in
SKIP_METHODS and the dump keeps the numbers. Which configurations the paper
shows is a figure decision and belongs here, not in ``results/``, which is
append-only input -- and a probe that has been run is worth keeping even when
it is not in the plot, because the argument for a figure is usually made with
the arm that was left out of it.

Only configurations that *have* a LAM appear here. The action-only policy has
no latent action model, so there is no latent space to probe; its row belongs
in the policy-performance table, not in this figure.

A file whose first line is ``# DUMMY DATA`` holds placeholders so the layout can
be reviewed before the runs finish, and the script prints a banner while that
line is there -- the same convention the other plot scripts in this repo use.
``results/probing_libero.csv`` is flagged that way right now: **no probe has been
run on plain LIBERO for any of these four configurations**, so its whole band is
made up. Every cell in it is a flat ``0.1111111`` -- four marks at one height in
every group, which is a shape no probe produces and which reads as placeholder
from across the room, without anyone having to compare decimals. Overwrite the
rows and delete the marker line when real numbers land; until then the figure is
for layout review only.

This script read that file as data until 2026-08-18, because it was the one plot
script here reading its CSV without ``comment="#"`` -- so the marker line would
have become the header and the banner never fired. An earlier round of filler
(a flat 0.1 in every cell, committed as "0.1 filler for the un-run cells") was
therefore drawn as a measurement. A convention only one script skips is the one
that bites.
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

# One figure per probe: which dumps to read, what to call the output, and what
# the y axis says. Both probes come out of the same probe job -- the analyzer is
# run with --probe-model both and its action_probe_r2.csv carries mlp and ridge
# rows over the identical train/val/test split -- so the two figures are paired
# cell for cell, not two experiments.
#
# WHY THE RIDGE DUMPS ARE A SUBDIRECTORY AND NOT results/probing_ridge_<suite>.csv.
# The default glob below is results/probing_*.csv, so a file named that way is
# picked up by a plain `python plot_probing.py` -- and because the ridge dumps
# carry the same axis rows and the same method columns, both consistency checks
# in main() would pass and the MLP figure would silently come out with six
# suite bands. A subdirectory cannot be globbed into the wrong figure.
#
# Adding a third probe is a row here plus a dump directory; nothing else in this
# file knows how many there are.
PROBES = {
    "mlp":   {"subdir": ".",             "name": "probing",       "label": "MLP"},
    "ridge": {"subdir": "probing_ridge", "name": "probing_ridge", "label": "ridge"},
}

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _r2 suffix stripped. Order here does not matter; the CSV column
# order is what fixes the plotting order and the marker assignment.
#
# Every configuration here is two-frame, Delta = [0,5], so the frame offsets are
# no longer a variable and have come out of the labels: what varies is the
# viewpoint count and what trains the latent. The two "Ours" rows are the same
# LAM at one and two viewpoints; CLAM, UniVLA and LAOF name the paper each
# borrowed component comes from, matching the policy table's wording so the
# figure and the table can be read against each other.
#
# This map covers every _r2 column the dumps carry, including the ones
# SKIP_METHODS holds back -- their labels are what the skip note prints.
#
# If a label ever carries math again, use brackets and not the set braces the
# paper uses for Delta. All figure text is one size (10pt), but Computer
# Modern's math brace \{ \} is a tall glyph built to wrap fractions: measured,
# "$1\times\{0,5\}$" renders 13.3pt against 9.5pt for every other label, so the
# legend reads as a larger font even though it is not. Brackets render at
# 11.8pt. (Text-mode braces do not work here either: matplotlib's raw cmr10 has
# no glyph at the ASCII { } slots, so "$1\times${0,5}" renders as garbage.)
METHOD_LABELS = {
    "sv_sf_r2": "Ours (single-view)",
    "mv_sf_r2": "Ours (multi-view)",
    "clam_r2": "CLAM-style (action grounding)",
    "dino_r2": "UniVLA-style (DINOv3 features)",
    "flow_r2": "LAOF-style (optical-flow decoder)",
}

# Columns present in the dumps but not drawn. Named by CSV column, so the
# entries here and the keys in METHOD_LABELS are the same vocabulary.
#
# A denylist rather than a list of what to plot, because the plotting order and
# the marker assignment come from the CSV column order -- an allowlist would
# quietly become a second place that decides the order, and the two would drift.
# It also keeps the "adding a configuration is a schema change only" property:
# a new _r2 column appears in the figure without anyone editing this file.
#
# The skipped columns are still read, cross-checked against their mean row, and
# named on stdout, so a column cannot vanish from the figure silently.
#
# dino_r2 (UniVLA-style, DINOv3 features) is out of the paper's probing figure
# as of 2026-09-02. Its numbers stay in the dumps: LIBERO-Plus mean 0.6212 and
# MimicGen 0.5171 are the standing evidence that probe R^2 does not rank these
# arms by success rate -- it probes above our single-view reference and scores
# below it -- which is why results/README.md still cites the column.
SKIP_METHODS = {"dino_r2"}


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
    """Our own configurations, flagged by "ours" in the display label. Kept in
    one place so the red-edge highlight in the plot and the legend never drift.

    Matched case-insensitively on the bare word rather than on "(ours)", so a
    label can lead with it ("Ours (single-view)") instead of trailing it. Note
    this is now true of two of the four series, not one: the highlight says
    "this is ours", and the shape still says which one."""
    return "ours" in label_for(method).lower()


def load(csv_path):
    """Return (axis labels, {method: values}, methods, skipped, n_placeholder,
    is_dummy).

    ``methods`` is the drawn subset, in CSV column order; ``skipped`` is what
    SKIP_METHODS held back, so the caller can say so out loud. Both are keyed on
    the CSV column name.

    Values are NaN wherever the dump carried the placeholder, so downstream
    code never has to special-case it -- matplotlib skips NaN, and nanmean
    ignores it.
    """
    # The dummy marker is a comment line, not a column, so it survives being
    # read as a CSV -- but only if the reader is told to skip comments, which
    # this one was not until 2026-08-18. Without comment="#" the marker becomes
    # the header row and a file of placeholders reads as a file of data.
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")
    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_r2")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_r2'")

    is_mean = df["action_dim"].astype(str).str.lower() == "mean"
    dims, mean_rows = df[~is_mean], df[is_mean]

    skipped = [m for m in methods if m in SKIP_METHODS]

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

    # Drawn columns only from here on: the placeholder count is reported as
    # "drawn as gaps", which is only true of a column that is drawn at all.
    methods = [m for m in methods if m not in SKIP_METHODS]
    if not methods:
        raise SystemExit(f"{csv_path.name}: SKIP_METHODS holds back every "
                         f"_r2 column -- nothing left to plot")
    values = {m: v for m, v in values.items() if m in methods}

    n_placeholder = sum(int((v == PLACEHOLDER).sum()) for v in values.values())
    values = {m: np.where(v == PLACEHOLDER, np.nan, v) for m, v in values.items()}
    return labels, values, methods, skipped, n_placeholder, is_dummy


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


def build_legends(fig, suite_labels, suite_cidx, methods, left):
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
        Patch(facecolor=band_colour(suite_cidx[i]), edgecolor="none", label=s)
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
                   help="specific result CSVs (default: every probing_*.csv in "
                        "the chosen probe's dump directory)")
    p.add_argument("--probe", choices=list(PROBES), default="mlp",
                   help="which probe's R^2 to draw: mlp (default, results/) or "
                        "ridge (results/probing_ridge/). Same jobs, same rows, "
                        "same figure -- one metric.")
    p.add_argument("--name", default=None,
                   help="output basename in figures/ (default: the chosen "
                        "probe's, plus _science under --style science)")
    p.add_argument("--style", choices=["paper", "science"], default="paper",
                   help="paper: exact IEEEtran/Computer-Modern match (default). "
                        "science: the SciencePlots aesthetic (garrettj403/SciencePlots).")
    args = p.parse_args()

    # The style is a swappable module exposing the same names (PALETTE, MARKERS,
    # apply_style, save, ...). Rebinding the module-global `style` here means the
    # draw functions, which look it up at call time, pick up whichever was asked
    # for without any per-call plumbing.
    global style
    probe = PROBES[args.probe]
    # An explicit --name is taken as given, including under --style science: the
    # caller who named the file is the one who has to keep the two apart.
    name = args.name or probe["name"]
    if args.style == "science":
        import style_science
        style = style_science
        if args.name is None:  # keep the two styles' outputs side by side
            name += "_science"

    dumps = (RESULTS_DIR / probe["subdir"]).resolve()
    paths = args.csv or sorted(dumps.glob("probing_*.csv"))
    if not paths:
        raise SystemExit(f"no probing CSVs found in {dumps}")
    order = {s: i for i, s in enumerate(SUITE_ORDER)}
    paths = sorted(paths, key=lambda q: (order.get(suite_from(q)[1], len(order)), q.stem))

    style.apply_style()
    import matplotlib.pyplot as plt

    print(f"\n  {probe['label']} probe, per-dimension R^2 "
          f"-- {dumps.relative_to(RESULTS_DIR.parent)}/")

    suites, total_placeholder, dummy, skipped = [], 0, [], []
    for path in paths:
        suite_label, stem = suite_from(path)
        labels, values, methods, skip, n_ph, is_dummy = load(path)
        total_placeholder += n_ph
        skipped += [m for m in skip if m not in skipped]
        if is_dummy:
            dummy.append(path.name)
        print_table(suite_label, labels, values, methods)
        suites.append((suite_label, labels, values, methods, stem))

    # Every suite must agree on the axis rows and the method columns, or the
    # shared x groups and the shared marker legend would both be lying.
    ref_labels, ref_methods = suites[0][1], suites[0][3]
    for suite_label, labels, _, methods, _ in suites[1:]:
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

    # Derived from the pitch, not fixed: adding or dropping a suite changes how
    # much room a mark has, and a constant tuned for one suite count is either
    # cramped or needlessly small at another. 0.72 is what the hand-tuned pair
    # came to (3.6pt on the 5.0pt pitch of three suites x four methods), so that
    # case is reproduced exactly and the others scale off it. The ceiling is the
    # legend's own mark size: a plot mark bigger than its key reads as a
    # different series. The floor is where the four shapes stop being distinct.
    MARKSIZE = min(4.8, max(3.0, 0.72 * pitch_pt))
    if pitch_pt < MARKSIZE:
        print(f"\n  ! marks are {MARKSIZE:.2f}pt on a {pitch_pt:.2f}pt pitch -- they "
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
    # A suite's colour is its position in SUITE_ORDER, not its position in this
    # run's file list, so dropping a suite from the figure does not recolour the
    # ones that stay. SUITE_ORDER only "fixes the color assignment" if it is read
    # that way -- indexing by file order silently repaints LIBERO-PLUS the moment
    # LIBERO leaves.
    cidx = [SUITE_ORDER.index(t[4]) if t[4] in SUITE_ORDER else len(SUITE_ORDER) + k
            for k, t in enumerate(suites)]

    for gi in range(n_groups):
        for si, (b0, b1) in enumerate(bands):
            ax.axvspan(x[gi] + b0, x[gi] + b1, color=band_colour(cidx[si]),
                       linewidth=0, zorder=0)

    for si, (_, _, values, methods, _) in enumerate(suites):
        for mi, m in enumerate(methods):
            ours = is_ours(m)
            ax.plot(
                x + slots[si][mi], values[m],
                linestyle="none",
                # `% len(MARKERS)` wraps rather than raises, and MARKERS holds
                # four shapes. Shape is the only channel a mark has here, so a
                # fifth drawn series silently reuses the first one's: between
                # 2026-08-28 and 2026-09-02 the dumps carried five columns and
                # mv_sf drew as CLAM's circle, separated from it only by the red
                # highlight this figure says is not the identity channel. The
                # pitch warning above does not catch it (4.01pt pitch against
                # 3.00pt marks, so nothing overlapped). SKIP_METHODS is what
                # holds the count at four today -- emptying it is one line, so
                # add a fifth shape in style.py before drawing a fifth series.
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
    ax.set_ylabel(rf"{probe['label']} probe $R^2$")

    build_legends(fig, [s[0] for s in suites], cidx, ref_methods, left)
    style.save(fig, name)
    # Said out loud, every run: a column that is in the dump but not in the
    # figure is exactly the thing a reader of the figure cannot see.
    if skipped:
        print("\n  not plotted, held back by SKIP_METHODS -- still in the dumps:")
        for m in skipped:
            print(f"      {m}  {label_for(m)}")
    if total_placeholder:
        print(f"\n  note: {total_placeholder} placeholder cell(s) not yet run; "
              f"drawn as gaps, not as zeros")
    if dummy:
        print("\n  " + "!" * 66)
        for n in dummy:
            print(f"  ! {n} is flagged DUMMY DATA -- these are")
        print("  ! placeholders, not measurements. Do not ship this figure.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
