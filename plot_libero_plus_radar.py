r"""LIBERO-Plus success rate, one spoke per disturbance dimension.

Reads ``results/libero_plus_radar.csv`` and emits ``figures/libero_plus_radar.pdf``:

    python plot_libero_plus_radar.py

Encoding: one spoke per disturbance dimension, and **colour + dash + marker
shape** together identify the arm. One factor, so all three channels are spent
on it -- the same choice plot_libero_radar.py and plot_realworld.py make, and
the opposite of probing.pdf, which has two factors and gives each its own
channel.

RELATIONSHIP TO plot_libero_radar.py. That script is the deliberate sibling of
this one and the two are independent copies, not a shared module. They plot
different experiments -- LIBERO suites under a held-out split there, LIBERO-Plus
disturbance dimensions here -- and each owns its own axes, labels, LaTeX snippet
and caption. The cost of the copy is that a layout fix made in one has to be
made in the other by hand; the placement code below is the part where that
matters, so read both before changing either.

What the two DO share, and must keep sharing, is the appearance of an arm. The
colour, dash and marker are assigned by column position, so a dump that puts the
weakest arm first and the ceiling last draws that arm red/solid/circle and that
ceiling green/dotted/triangle in both figures. Both dumps do. Reorder this
file's ``_sr`` columns and the same policy changes appearance between two
figures in the same paper.

WHY A RADAR AT ALL. Five dimensions x three arms is a small grouped bar chart's
job, and a bar chart reads values more accurately. The radar earns its place
here for the same reason it does on the LIBERO suites: the *shape* is the
finding. The oracle-latent polygon is close to round -- full data is roughly
uniformly robust across disturbance types -- while both two-demonstration
polygons are pulled sharply in at `camera` and `noise`. "Which disturbances does
the latent action model actually buy robustness to, and which does it not" is
the reading a reader takes away, and it is a shape, not five numbers. Do not add
a fourth arm: past three polygons the fills occlude each other and the shape
reading is gone, which is the only reason the form was chosen. Exact values live in the printed table and
in the paper's tables, never in the figure.

Radar caveats the caption should not pretend away: area scales as the square of
the radius, so a polygon at 70 looks nearly twice as "big" as one at 50; and the
spoke order is arbitrary, so the polygon's shape changes if the dimensions are
reordered. DIM_ORDER is therefore fixed, and it is the dump's own order rather
than anything derived from the values -- sorting the spokes by gain would make
the shape a function of the numbers and it would change under every new sweep.

THE AGGREGATE GETS A SPOKE, as it does in plot_libero_radar.py, so the two
figures read the same way. It comes last, which puts it adjacent to the top
spoke where it reads as the summary rather than as a sixth disturbance.

It is labelled "All (mean)", the same as on the LIBERO radars, and like them it
plots the value the dump carries rather than one this script computes.

Be precise about what that value is. Here the dump's `all` row is the rate
pooled over all 1,942 episodes, and the dimensions do not hold equal numbers of
episodes, so it is not literally the mean of the five spokes -- on this sweep
the mean is 49.96 / 62.30 / 85.15 against a pooled 49.79 / 62.10 / 85.12. A
fifth of a point is well inside MEAN_ATOL and invisible at this radius, which is
what makes the shared label honest; the tolerance check is what keeps it that
way. The paper's tables quote the pooled number, and plotting the same value the
tables quote is the invariant that matters. If a future sweep weights the
dimensions unevenly the check will warn, and then both the caption and this
label have to be revisited.

Six spokes is an EVEN count, which puts a spoke label at due south -- where the
legend goes. That is a real layout hazard and the reason the five-dimension
version of this figure left the aggregate off. It is handled in draw() by
reserving a band for that one label and pushing the legend below it, so nothing
overprints; the cost is that this figure is ~0.2in taller than the LIBERO
radars. See SPOKE_LABEL_IN and the margin derivation there before changing the
spoke count again.

PALETTE. This figure deliberately does NOT use style.PALETTE, for the reason
given at length in plot_libero_radar.py: the draft's radars were already
red/blue/green and the generated ones match them so the paper does not show two
colour schemes for the same arms. red/green is the classic dichromat collision
and style.py's derivation does not cover it. What keeps the figure readable is
that hue is not carrying identity alone -- every polygon also has its own dash
pattern (solid / dashed / dotted) and its own marker shape (o / s / ^), both of
which survive deuteranopia and a grayscale photocopy. Drop the dashes or the
markers and this figure has to move to style.PALETTE first.

EMBEDDING IN LATEX
------------------
Built at 3.487in = \columnwidth (252pc, per IEEEtran journal mode), so this is a
plain figure, not a figure*. Needs \usepackage{graphicx}.

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/libero_plus_radar.pdf}
      \caption{LIBERO-Plus success rate per disturbance dimension, over a fixed
      sweep selection of 1{,}942 episodes evaluated identically for every arm.
      \emph{Action-only} and \emph{Learned latent} both see two action-labelled
      demonstrations per task, with and without the single-view latent action
      model; \emph{Oracle latent} is the ceiling, trained on all 15{,}446 Panda
      demonstrations. The latent action model recovers most of the gap under
      \emph{add\_object} and \emph{camera} but little of it under \emph{noise},
      where the oracle stays near the rim and both two-demonstration polygons
      collapse. \emph{All (mean)} is the rate pooled over all 1{,}942 episodes:
      $49.8 \rightarrow 62.1$ against a ceiling of $85.1$.}
      \label{fig:libero_plus_radar}
    \end{figure}

Include it at \columnwidth and nothing is rescaled. Any other width -- including
width=0.7\columnwidth -- scales the figure text off its point size, which is the
one thing the README forbids.

TEXT SIZE. Like the two LIBERO radars and unlike the repo's other figures, this
is set at 10pt to match the body text rather than at style.py's 8pt -- see
RADAR_PT below for what that costs and how to undo it. It only holds at
width=\columnwidth.

The layout is tight at that size and it is tight on purpose: at 10pt these
three labels come to 3.430in of a 3.487in column -- the same content as
plot_libero_radar.py's legend, so the two frames land at the same width without
either script forcing it. That leaves almost no headroom. Lengthening a label,
or adding a fourth arm, will push the legend off the page. If that happens,
shrink the legend spacings before touching the margins -- the margins are what
the circle's diameter is made of.

Input schema (see results/README.md). One row per disturbance dimension, plus an
optional aggregate row where ``dim == "all"``:

    dim,<method>_sr,<method>_sr,...

Any column ending in ``_sr`` is treated as an arm, in file order. Success rates
are stored as fractions in [0, 1] and converted to percent here; the axis is the
only place the number is a percentage.

As in the other success-rate dumps, 0 is a REAL value: an arm that never
survives a disturbance genuinely scores 0%. A run that has not happened is left
EMPTY, which reads as NaN and breaks the polygon at that spoke rather than
pulling it to the centre.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "libero_plus_radar.csv"

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _sr suffix stripped. Order here does not matter; the CSV column
# order fixes the plotting order, the colour, the dash and the marker.
#
# Verbatim from plot_libero_radar.py's METHOD_LABELS. The two figures show the
# same three arms and a reader meets them one page apart, so they are named the
# same thing in both -- calling the middle one "Learned latent" here and
# "N=2 + LAM" there would read as two different methods. Keep them in sync: if
# that file's labels are renamed, rename these too.
#
# The column NAMES stay experiment-specific (n2_sr, ceiling_sr) because the dump
# is the record of what was run; only the display names are shared.
#
# What each arm actually is -- two action-labelled demos per task, the LAM being
# single-view, the ceiling's 15,446 Panda demonstrations -- belongs in the
# caption, which is also where the two figures' arms stop being interchangeable.
METHOD_LABELS = {
    "n2_sr": "Action-only",
    "n2_lam_sr": "Learned latent",
    "ceiling_sr": "Oracle latent",
}

DIM_LABELS = {
    "add_object": "Add object",
    "camera": "Camera",
    "lighting": "Lighting",
    "texture": "Texture",
    "noise": "Noise",
    # Verbatim from plot_libero_radar.py's SUITE_LABELS, so the aggregate spoke
    # is named the same thing on every radar in the paper.
    #
    # Read MEAN_ATOL before trusting the word "mean" here: the value plotted is
    # the dump's `all` row, which for this experiment is pooled over episodes
    # rather than averaged over the five dimensions. The two agree to 0.2 points
    # on this sweep and the tolerance check is what keeps them agreeing, so the
    # label holds. If that check ever starts warning, this label is the second
    # thing to fix after the dump.
    "all": "All (mean)",
}

# Spoke order, clockwise from the top. A radar's shape is an artifact of this
# order, so it is fixed here -- reordering it would change the polygon in the
# paper without changing a single number. This is the dump's own column order
# and nothing is derived from the values: sorting the spokes by, say, the LAM
# gain would make the figure's shape a function of the numbers, so it would
# change silently under the next sweep and stop being comparable to this one.
DIM_ORDER = ["add_object", "camera", "lighting", "texture", "noise"]

# Drawn as a sixth spoke, last, so it sits adjacent to the top one and reads as
# the summary rather than as another disturbance -- the same placement
# plot_libero_radar.py gives its aggregate. Six spokes is an even count and puts
# a label at due south; draw() reserves a band for it. See the docstring.
AGGREGATE = "all"

# Drawn inside the figure. Unlike plot_libero_radar.py's two panels this figure
# has no sibling to be told apart from, so the title names the benchmark rather
# than the split.
PANEL_TITLE = "LIBERO-Plus Disturbances"

# The aggregate row is cross-checked against the mean over the dimension rows.
# Same tolerance as plot_probing.py uses for its `mean` row. Note that here the
# two are genuinely different quantities -- the dump's `all` is pooled over
# episodes, not a mean of the five dimensions -- so this is a loose sanity check
# on the dump, not an identity. See the header comment in the CSV.
MEAN_ATOL = 5e-3

# See the PALETTE note in the module docstring. Positional, like style.PALETTE,
# but local to this figure, and index-matched to plot_libero_radar.py's copy so
# an arm in the same column position looks the same in both figures.
#
# DASHES and RADAR_MARKERS are not decoration. They are the second and third
# channels that keep the figure legible for a red/green dichromat and in a
# grayscale photocopy, which this palette on its own does not. Every entry here
# is index-matched to RADAR_PALETTE and they must stay that way.
RADAR_PALETTE = ["#E31A1C", "#1F77B4", "#2CA02C"]
DASHES = ["-", "--", ":"]
RADAR_MARKERS = ["o", "s", "^"]

# Polygon fill, per series. Low enough that three stacked fills still let the
# innermost polygon's outline read; the fill is there to make the enclosed
# envelope legible as an area, not to identify the series -- that is what the
# outline does.
FILL_ALPHA = 0.10

# Clearance between the rim and the nearest edge of a spoke label, in points.
# Real clearance, not a radial offset -- see the placement code in draw().
SPOKE_PAD_PT = 4.0

# Layout reserves, in inches. draw() derives every margin from these rather than
# carrying hard-coded figure fractions, so the one number that changes with the
# spoke count -- whether a label lands at due south -- flows through to the
# figure height, the bottom margin and the legend anchor together instead of
# having to be re-tuned in three places.
#
#   SIDE_IN         the widest spoke labels, which stick out further than
#                   anything on the top or bottom. Generous on purpose: the two
#                   left spokes sit where the rim curves *towards* the label, so
#                   a radial pad that clears the arc at its tangent point still
#                   lets a corner of the text cross it further along. Buying the
#                   clearance in the margin rather than by shrinking SPOKE_PAD_PT
#                   is what keeps the labels off the circle.
#   TOP_IN          panel title, plus the top spoke label below it
#   LEGEND_GAP_IN   circle (or the south label, when there is one) to legend top
#   LEGEND_ROW_IN   the framed legend itself, one row of three entries
#   SPOKE_LABEL_IN  one line of spoke label plus SPOKE_PAD_PT, reserved below
#                   the circle ONLY when an even spoke count puts a label at due
#                   south. At an odd count this is zero and the derivation below
#                   collapses to exactly the margins the LIBERO radars use.
SIDE_IN = 0.70
TOP_IN = 0.48
LEGEND_GAP_IN = 0.125
LEGEND_ROW_IN = 0.345
SPOKE_LABEL_IN = 0.20


# Figure text size for THIS figure only, in TeX points. style.py sets 8pt
# (IEEEtran \footnotesize) for the repo's figures; the three radars are set at
# the 10pt body size instead, by explicit request, so their labels match the
# surrounding text rather than sitting a step below it.
#
# This is a local deviation and it does cost something: the paper now carries
# two figure text sizes, and a reader flipping between probing.pdf and a radar
# will see it. If the rest of the figures ever move to 10pt, delete this block
# and set FIG_PT_TEX = 10 in style.py, which is the knob built for it.
#
# The 72/72.27 conversion is the same one style.py derives: a size in TeX
# points is not a matplotlib font size, which is in PostScript points.
RADAR_PT_TEX = 10
RADAR_PT = RADAR_PT_TEX * 72 / 72.27

# Radial grid. Rings at these percentages, plus the rim at 100.
RINGS = [25, 50, 75, 100]

# Where the radial tick labels sit, in degrees: the midpoint of the wedge
# between the first and second spokes, so the 25/50/75/100 column runs up empty
# space instead of sitting on a spoke label or across a polygon edge.
#
# NOT plot_libero_radar.py's 54, even though the two figures are otherwise
# matched. Six spokes are 60 degrees apart rather than 72, and at 54 the column
# lands almost on the "Camera" spoke -- "100" collided with its label and "50"
# crossed the blue polygon. Checked against 45 as well, which has the same
# problem more mildly. This value is a function of the spoke count; re-check it
# if DIM_ORDER ever changes length.
RLABEL_DEG = 30


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_sr") else method)


def load(csv_path):
    """Return (dimension keys, {method: values in %}, methods, is_dummy).

    The aggregate, when present, is appended after the dimensions so the printed
    table can report it; draw() is handed the dimension slice only.

    Empty cells read as NaN, which matplotlib skips -- so a partially filled
    dump plots the dimensions it has instead of failing or drawing zeros.
    """
    # The dummy marker is a comment line, not a column, so it survives being
    # read back by anything else that consumes this CSV.
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")

    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_sr'")

    df = df.set_index(df["dim"].astype(str).str.strip())
    missing = [d for d in DIM_ORDER if d not in df.index]
    if missing:
        raise SystemExit(f"{csv_path.name}: no row for dimension(s) "
                         f"{', '.join(missing)}; the spokes are fixed by "
                         f"DIM_ORDER so the axes cannot silently change")

    dims = list(DIM_ORDER)
    values = {m: df.loc[dims, m].to_numpy(dtype=float) for m in methods}

    if AGGREGATE in df.index:
        dims.append(AGGREGATE)
        for m in methods:
            # Sanity-check the dumped aggregate against the mean over the
            # dimensions. These are not the same quantity (see MEAN_ATOL), but
            # they cannot be far apart unless the dump is stale.
            dumped = float(df.loc[AGGREGATE, m])
            recomputed = np.nanmean(values[m])
            if not np.isnan(recomputed) and not np.isclose(
                    dumped, recomputed, atol=MEAN_ATOL):
                print(f"  ! {csv_path.name}: {m} '{AGGREGATE}' row is "
                      f"{dumped:.4f}, mean over dimensions is {recomputed:.4f}")
            values[m] = np.append(values[m], dumped)

    for m in methods:
        if np.nanmax(values[m], initial=0.0) > 1.0:
            raise SystemExit(
                f"{csv_path.name}: {m} exceeds 1.0 -- success rates are stored "
                f"as fractions in [0, 1] and converted to percent by this script")
        values[m] = values[m] * 100.0

    return dims, values, methods, is_dummy


def print_table(dims, values, methods):
    """Text table view -- identity is never carried by colour alone, and this is
    also the fastest way to spot a stale dump.

    Printed at four decimals: enough to make the 10-decimal dummy placeholders
    obvious at a glance, since a real success rate over N episodes lands on a
    round fraction.

    This is also the only place the aggregate row appears, since it gets no
    spoke. The figure deliberately shows no numbers; this is where they live."""
    head = f"{'dimension':<14}" + "".join(f"{label_for(m):>18}" for m in methods)
    print(f"\n  {PANEL_TITLE}  (success rate, %)")
    print("  " + head)
    print("  " + "-" * len(head))
    for i, d in enumerate(dims):
        if d == AGGREGATE:
            print("  " + "-" * len(head))
        cells = "".join(
            f"{values[m][i]:>18.4f}" if not np.isnan(values[m][i]) else f"{'--':>18}"
            for m in methods)
        print("  " + f"{DIM_LABELS.get(d, d):<14}" + cells)


def draw(dims, values, methods, name):
    """Draw the radar and write figures/<name>.pdf.

    `dims` are all the spokes, aggregate included. An even count reserves a band
    below the circle for the label that lands at due south -- see `band` below.
    """
    import matplotlib.pyplot as plt

    n = len(dims)
    # One angle per spoke, clockwise from the top (set_theta_* below). The first
    # angle is repeated at the end so every polygon closes; the same trick is
    # applied to the values.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(angles, angles[0])

    # An even spoke count puts a label at due south, where the legend goes --
    # with six spokes "Texture" lands straight down. Rather than warn and let
    # them overprint, reserve a band for that one label and push the legend
    # below it. At an odd count the band is zero and every number below comes
    # out at the LIBERO radars' hand-tuned values.
    band = SPOKE_LABEL_IN if n % 2 == 0 else 0.0

    # Explicit margins, not constrained_layout, because savefig.bbox is off (see
    # style.py): the saved page has to be exactly figsize or the include
    # rescales the text. Derived in inches from the reserves above and converted
    # here, so the reserved space stays fixed if the height ever changes.
    #
    # The band is added to the figure HEIGHT as well as to the bottom margin, so
    # the axes box -- and therefore the circle -- comes out the same size it
    # would at an odd spoke count. Taking it out of the margin alone would have
    # shrunk the circle instead, and the circle is the figure.
    #
    # The height is otherwise chosen so the axes box comes out roughly square,
    # i.e. no taller than the circle the side margins already fix the width of.
    # A taller box does not enlarge the circle -- a polar axes inscribes it --
    # it just adds white above and below, and the legend does not fill that
    # white because matplotlib anchors it to the circle's bounding box rather
    # than to the axes box.
    w = style.COL_WIDTH
    bottom_in = LEGEND_GAP_IN + band + LEGEND_ROW_IN
    h = w * 0.87 + band
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(projection="polar")
    fig.subplots_adjust(left=SIDE_IN / w, right=1 - SIDE_IN / w,
                        bottom=bottom_in / h, top=1 - TOP_IN / h)

    # Axes height in inches, for the legend anchor below -- bbox_to_anchor is in
    # axes fractions, but the gap it has to clear is a physical length.
    axes_h_in = h - TOP_IN - bottom_in

    ax.set_theta_offset(np.pi / 2)   # first spoke straight up
    ax.set_theta_direction(-1)       # then clockwise, as read

    for mi, m in enumerate(methods):
        v = np.append(values[m], values[m][0])
        colour = RADAR_PALETTE[mi % len(RADAR_PALETTE)]
        ax.plot(closed, v,
                linestyle=DASHES[mi % len(DASHES)],
                linewidth=1.2,
                marker=RADAR_MARKERS[mi % len(RADAR_MARKERS)],
                markersize=3.6,
                color=colour,
                # No contrasting edge: at 3.6pt a 0.6pt contour is a sixth of
                # the mark, and these fills are saturated enough to carry
                # themselves against the near-white rings.
                markeredgecolor=colour,
                label=label_for(m),
                zorder=4 + mi)
        # fill(), not fill_between(): the polygon is closed in theta, and a NaN
        # spoke has to leave a hole rather than being bridged.
        #
        # linestyle is pinned even though the fill has no edge: under the
        # SciencePlots style the polygon inherits a dashed linestyle from the
        # prop cycle, and matplotlib scales the dash pattern by the linewidth,
        # so lw=0 turns it into [0, 0] and raises at draw time.
        ax.fill(closed, v, color=colour, alpha=FILL_ALPHA, linewidth=0,
                linestyle="solid", zorder=2 + mi)

    ax.set_ylim(0, 100)
    ax.set_yticks(RINGS)
    # Numbers on the rings, not on the data: these are the radial scale, the one
    # place this figure prints values. Grey and small so they sit under the
    # polygons rather than competing with them.
    #
    # Sized off rcParams rather than style.FIG_PT: FIG_PT exists in style.py but
    # not in style_science.py, and this script has to work under either.
    ax.set_yticklabels([str(r) for r in RINGS], color=style.INK_MUTED,
                       fontsize=plt.rcParams["font.size"] * 0.85)
    ax.set_rlabel_position(RLABEL_DEG)

    ax.set_xticks(angles)
    # Spoke labels are placed by hand, not left to tick_params(pad=...).
    # matplotlib pads a polar tick label radially but still anchors its *box* on
    # the arc, so on an angled spoke a corner of the box crosses the rim however
    # large the pad gets -- and growing the pad to hide that just pushes the
    # label off the page. Anchoring each label on the side facing the circle
    # makes its box grow outward instead, so a few points of pad is genuinely a
    # few points of clearance.
    ax.set_xticklabels([])
    for ang, key in zip(angles, dims):
        # Screen direction of this spoke, given the offset/direction set above:
        # theta is measured clockwise from straight up.
        ux, uy = np.sin(ang), np.cos(ang)
        ax.annotate(
            DIM_LABELS.get(key, key),
            xy=(ang, 100), xycoords="data",
            xytext=(SPOKE_PAD_PT * ux, SPOKE_PAD_PT * uy),
            textcoords="offset points",
            # "center" on the near-tangential axis, so a label beside the circle
            # stays visually centred on its spoke rather than being shunted a
            # half line-height up or down by a small perpendicular component.
            ha="center" if abs(ux) < 0.35 else ("left" if ux > 0 else "right"),
            va="center" if abs(uy) < 0.35 else ("bottom" if uy > 0 else "top"),
            annotation_clip=False,
        )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # Rings and spokes as light dashed rules, and the rim as a solid grey
    # circle: the rim is the 100% boundary, a real reference the reader reads
    # against, while the inner rings are only a scale. Drawing them the same
    # weight would make 75 look as meaningful as 100.
    ax.grid(color=style.GRID, linewidth=0.5, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color(style.INK_MUTED)
    ax.spines["polar"].set_linewidth(0.8)
    # Solid, set explicitly: the rim inherits a dashed pattern under the
    # SciencePlots style, which both blurs the 100% boundary into the inner
    # rings and raises "at least one value in the dash list must be positive"
    # at draw time.
    ax.spines["polar"].set_linestyle("solid")
    ax.set_facecolor("white")

    # Title placed by hand, not with set_title(). On a polar axes matplotlib
    # measures the title pad from the axes *box*, and the "Add object" spoke
    # label is drawn outside that box -- so every pad small enough to look right
    # overprints the label, and every pad large enough to clear it leaves a
    # hole. A figure-coordinate anchor in the margin reserved above is
    # deterministic and does not depend on what the top spoke is called.
    fig.text(0.5, 1 - 0.155 / h, PANEL_TITLE, ha="center", va="center",
             fontweight="bold")

    # Legend below the axes, in a frame. Boxed rather than the frameless default
    # of style.py: it sits outside the axes over open page here, and the frame
    # is what stops it reading as a caption fragment. One row, three entries,
    # which is what the three-polygon limit above buys.
    #
    # The spacings are tight because three entries on one row is very close to
    # the page width at column size: at the matplotlib defaults this legend
    # measured wider than the figure and ran off both edges. Widen anything here
    # -- or lengthen a METHOD_LABELS entry -- and re-check that the frame still
    # fits inside figsize.
    #
    # The anchor clears LEGEND_GAP_IN below the circle, plus the south spoke
    # label when there is one. Expressed as a physical length over the axes
    # height because bbox_to_anchor is in axes fractions: hard-coding the
    # fraction would silently change the gap whenever the band does.
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -(LEGEND_GAP_IN + band) / axes_h_in),
              ncol=len(methods), frameon=True, fancybox=False,
              edgecolor=style.INK_MUTED, framealpha=1.0,
              borderaxespad=0, handlelength=1.4, handletextpad=0.4,
              columnspacing=0.7, borderpad=0.35)
    leg = ax.get_legend()
    leg.get_frame().set_linewidth(0.6)

    return style.save(fig, name)


def main():
    # Raw formatter: the default one reflows paragraphs, which would collapse
    # the LaTeX snippet above into an unusable single block.
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                   help=f"result CSV (default: {DEFAULT_CSV.name})")
    p.add_argument("--style", choices=["paper", "science"], default="paper",
                   help="paper: exact IEEEtran/Computer-Modern match (default). "
                        "science: the SciencePlots aesthetic (garrettj403/SciencePlots).")
    args = p.parse_args()

    # The style is a swappable module exposing the same names (COL_WIDTH,
    # apply_style, save, ...). Rebinding the module-global `style` here means
    # the draw code, which looks it up at call time, picks up whichever was
    # asked for without any per-call plumbing. RADAR_PALETTE is not swapped --
    # it is this figure's own, see the docstring.
    global style
    suffix = ""
    if args.style == "science":
        import style_science
        style = style_science
        suffix = "_science"  # keep the two styles' outputs side by side

    if not args.csv.exists():
        raise SystemExit(f"no such results file: {args.csv}")

    style.apply_style()

    # Local font-size override, applied after the shared style so it wins over
    # whichever module was swapped in above. See RADAR_PT.
    import matplotlib.pyplot as plt
    plt.rcParams.update({k: RADAR_PT for k in (
        "font.size", "axes.labelsize", "axes.titlesize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})

    dims, values, methods, is_dummy = load(args.csv)
    print_table(dims, values, methods)
    # The aggregate is drawn as a spoke like the rest -- see AGGREGATE.
    draw(dims, values, methods, args.csv.stem + suffix)

    missing = sum(int(np.isnan(values[m]).sum()) for m in methods)
    if missing:
        print(f"\n  note: {missing} empty cell(s) not yet run; the polygon is "
              f"broken at those spokes, not pulled to zero")
    if is_dummy:
        print("\n  " + "!" * 66)
        print(f"  ! {args.csv.name} is flagged DUMMY DATA -- these are")
        print("  ! placeholders, not measurements. Do not ship this figure.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
