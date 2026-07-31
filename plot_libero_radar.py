r"""Per-suite LIBERO success rate, one radar per held-out / non-held-out split.

Reads every ``results/libero_radar_<split>.csv`` and emits one figure per file,
``figures/libero_radar_<split>.pdf``. Run with no arguments to rebuild both:

    python plot_libero_radar.py

Encoding: one spoke per LIBERO suite plus the aggregate, and **colour + dash +
marker shape** together identify the policy. One factor, so all three channels
are spent on it -- the same choice plot_realworld.py makes, and the opposite of
probing.pdf, which has two factors and gives each its own channel.

WHY A RADAR AT ALL. Four suites x three policies is a small grouped bar chart's
job, and a bar chart reads values more accurately. The radar earns its place
here only because the *shape* is the finding: on held-out tasks the oracle
polygon stays near the rim while the learned one collapses inward, and that
"how much of the achievable envelope did we recover" reading is what a reader
takes away. Do not add a fifth or sixth policy -- past three polygons the fills
occlude each other and the shape reading is gone, which is the only reason the
form was chosen. Exact values live in the printed table and in the paper's
tables, never in the figure.

Radar caveats the caption should not pretend away: area scales as the square of
the radius, so a polygon at 70 looks nearly twice as "big" as one at 50; and the
spoke order is arbitrary, so the polygon's shape changes if the suites are
reordered. SUITE_ORDER is therefore fixed and shared by both splits.

PALETTE. This figure deliberately does NOT use style.PALETTE. The two radars it
replaces were already in the draft in red/blue/green, and matching them was an
explicit call so the new generated figures drop into the paper without the
reader seeing two different colour schemes for the same three policies. The cost
is real and worth stating: red/green is the classic dichromat collision, and
style.py's palette derivation -- which checks all pairs under deuteranopia --
does not cover these colours. What keeps the figure readable anyway is that hue
is not carrying identity alone here: every polygon also has its own dash pattern
(solid / dashed / dotted) and its own marker shape (o / s / ^), both of which
survive deuteranopia and a grayscale photocopy. If you ever drop the dashes or
the markers, this figure stops being accessible -- switch it to style.PALETTE
first. See RADAR_PALETTE below.

EMBEDDING IN LATEX
------------------
Built at 3.487in = \columnwidth (252pc, per IEEEtran journal mode), so each is a
plain figure, not a figure*. Needs \usepackage{graphicx}.

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/libero_radar_nonh.pdf}
      \caption{Per-suite LIBERO task success breakdown for the 5\% action
      budget, on tasks that receive action-labelled demonstrations. Each spoke
      is one LIBERO suite; \emph{All (avg)} is the mean over the four.}
      \label{fig:libero_radar_nonh}
    \end{figure}

    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/libero_radar_h.pdf}
      \caption{Per-suite LIBERO task success breakdown for the 5\% action budget
      on unseen tasks (depth sweep). Same axes as
      Fig.~\ref{fig:libero_radar_nonh}; the oracle latent stays near the rim
      while the learned latent collapses inward.}
      \label{fig:libero_radar_h}
    \end{figure}

This REPLACES the two \includegraphics[width=0.7\columnwidth]{images/
libero_radar_NonH.pdf} / {images/libero_radar_H.pdf} lines in main.tex. Both the
path and the width change: 0.7\columnwidth scaled the text off its point size,
which is the one thing the README forbids. These are built at exactly
\columnwidth, so include them at \columnwidth and nothing is rescaled.

TEXT SIZE. Unlike the repo's other figures, these two are set at 10pt to match
the body text rather than at style.py's 8pt -- see RADAR_PT below for what that
costs and how to undo it. It only holds at width=\columnwidth; any other width
scales it off 10pt again.

The layout is tight at that size and it is tight on purpose: at 10pt the legend
is about 3.4in of a 3.487in column, and the "All (mean)" spoke label clears the
page edge by roughly 2pt. Renaming a method to something longer, or adding a
fourth, will push one of them off the page. If that happens, shrink the legend
spacings before touching the margins -- the margins are what the circle's
diameter is made of.

Input schema (see results/README.md). One row per suite, plus an optional
aggregate row where ``suite == "all"``:

    suite,<method>_sr,<method>_sr,...

Any column ending in ``_sr`` is treated as a policy, in file order, so the
colour, dash and marker assignment follow the column order -- keep the two
splits' columns in the same order or a policy changes colour between the two
figures. Success rates are stored as fractions in [0, 1] and converted to
percent here; the axis is the only place the number is a percentage.

As in the real-world dumps, 0 is a REAL value: a policy that never solves a
held-out suite genuinely scores 0%. A run that has not happened is left EMPTY,
which reads as NaN and breaks the polygon at that spoke rather than pulling it
to the centre.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import style

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Display names, keyed by CSV column. Unknown columns fall back to the column
# name with the _sr suffix stripped. Order here does not matter; the CSV column
# order fixes the plotting order, the colour, the dash and the marker.
METHOD_LABELS = {
    "action_only_sr": "Action-only",
    "learned_sr": "Learned latent",
    "oracle_sr": "Oracle latent",
}

SUITE_LABELS = {
    "object": "Object",
    "spatial": "Spatial",
    "goal": "Goal",
    "libero_10": "LIBERO-10",
    "all": "All (mean)",
}

# Spoke order, clockwise from the top. A radar's shape is an artifact of this
# order, so it is fixed here and shared by both splits -- reordering it would
# change every polygon in the paper without changing a single number. The
# aggregate always comes last, so it lands adjacent to the top spoke and reads
# as the summary rather than as a fifth suite.
SUITE_ORDER = ["object", "spatial", "goal", "libero_10"]
AGGREGATE = "all"

# Panel titles, drawn inside the figure. The original placeholder images carried
# an "(a)" / "(b)" prefix here, dropped because these are two separate floats
# with their own captions: LaTeX numbers them "Fig. 5" and "Fig. 6", and a
# figure captioned "Fig. 6" with "(b)" printed inside it refers to nothing.
# Put the prefixes back only if both radars are ever wrapped in one figure with
# subcaptions, and then let \subcaption set them rather than this dict, so the
# letters cannot disagree with the float they sit in.
PANEL_TITLES = {
    "h": "Held-Out Tasks",
    "nonh": "Non-Held-Out Tasks",
}

# Splits are drawn in this order when several are built; anything else follows,
# sorted. Only affects the order the tables print in.
SPLIT_ORDER = ["h", "nonh"]

# The aggregate row is cross-checked against the mean over the suite rows. Same
# tolerance as plot_probing.py uses for its `mean` row, and for the same reason:
# a disagreement above this is almost always a stale dump, not rounding.
MEAN_ATOL = 5e-3

# See the PALETTE note in the module docstring. Positional, like style.PALETTE,
# but local to this figure: these are the colours the draft's placeholder radars
# already used, kept so the generated figures are drop-in replacements.
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

# Figure text size for THIS figure only, in TeX points. style.py sets 8pt
# (IEEEtran \footnotesize) for the repo's figures; these two radars are set at
# the 10pt body size instead, by explicit request, so their labels match the
# surrounding text rather than sitting a step below it.
#
# This is a local deviation and it does cost something: the paper now carries
# two figure text sizes, and a reader flipping between probing.pdf and these
# will see it. If the rest of the figures ever move to 10pt, delete this block
# and set FIG_PT_TEX = 10 in style.py, which is the knob built for it.
#
# The 72/72.27 conversion is the same one style.py derives: a size in TeX
# points is not a matplotlib font size, which is in PostScript points.
RADAR_PT_TEX = 10
RADAR_PT = RADAR_PT_TEX * 72 / 72.27

# Radial grid. Rings at these percentages, plus the rim at 100.
RINGS = [25, 50, 75, 100]

# Where the radial tick labels sit, in degrees. Halfway between the first and
# second spokes, so the 25/50/75/100 column runs up the empty wedge instead of
# sitting on a spoke label or under a polygon vertex.
RLABEL_DEG = 54


def split_from(csv_path):
    """Split key parsed out of the filename, e.g. libero_radar_h.csv -> "h"."""
    return re.sub(r"^libero_radar_", "", csv_path.stem)


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_sr") else method)


def load(csv_path):
    """Return (suite keys, {method: values in %}, methods, is_dummy).

    Empty cells read as NaN, which matplotlib skips -- so a partially filled
    dump plots the suites it has instead of failing or drawing zeros.
    """
    # The dummy marker is a comment line, not a column, so it survives being
    # read back by anything else that consumes this CSV.
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")

    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_sr'")

    df = df.set_index(df["suite"].astype(str).str.strip())
    missing = [s for s in SUITE_ORDER if s not in df.index]
    if missing:
        raise SystemExit(f"{csv_path.name}: no row for suite(s) "
                         f"{', '.join(missing)}; the spokes are fixed by "
                         f"SUITE_ORDER so every split draws the same axes")

    suites = list(SUITE_ORDER)
    values = {m: df.loc[suites, m].to_numpy(dtype=float) for m in methods}

    if AGGREGATE in df.index:
        suites.append(AGGREGATE)
        for m in methods:
            # Check the dumped aggregate against the mean over the suites before
            # trusting it. A mismatch usually means a stale dump.
            dumped = float(df.loc[AGGREGATE, m])
            recomputed = np.nanmean(values[m])
            if not np.isnan(recomputed) and not np.isclose(
                    dumped, recomputed, atol=MEAN_ATOL):
                print(f"  ! {csv_path.name}: {m} '{AGGREGATE}' row is "
                      f"{dumped:.4f}, mean over suites is {recomputed:.4f}")
            values[m] = np.append(values[m], dumped)

    for m in methods:
        if np.nanmax(values[m], initial=0.0) > 1.0:
            raise SystemExit(
                f"{csv_path.name}: {m} exceeds 1.0 -- success rates are stored "
                f"as fractions in [0, 1] and converted to percent by this script")
        values[m] = values[m] * 100.0

    return suites, values, methods, is_dummy


def print_table(title, suites, values, methods):
    """Text table view -- identity is never carried by colour alone, and this is
    also the fastest way to spot a stale dump.

    Printed at four decimals: enough to make the 10-decimal dummy placeholders
    obvious at a glance, since a real success rate over N rollouts lands on a
    round fraction.

    The figure deliberately shows no numbers; this is where they live."""
    head = f"{'suite':<12}" + "".join(f"{label_for(m):>18}" for m in methods)
    print(f"\n  {title}  (success rate, %)")
    print("  " + head)
    print("  " + "-" * len(head))
    for i, s in enumerate(suites):
        if s == AGGREGATE:
            print("  " + "-" * len(head))
        cells = "".join(
            f"{values[m][i]:>18.4f}" if not np.isnan(values[m][i]) else f"{'--':>18}"
            for m in methods)
        print("  " + f"{SUITE_LABELS.get(s, s):<12}" + cells)


def draw(split, suites, values, methods, name):
    """Draw one radar and write figures/<name>.pdf."""
    import matplotlib.pyplot as plt

    n = len(suites)
    # One angle per spoke, clockwise from the top (set_theta_* below). The first
    # angle is repeated at the end so every polygon closes; the same trick is
    # applied to the values.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(angles, angles[0])

    # The bottom margin holds the legend, and it is only free because an odd
    # spoke count puts no label at due south -- with five spokes the two lower
    # labels sit 36 degrees off it. An even count (or dropping the aggregate
    # row from a four-suite dump) puts a label straight down, underneath the
    # legend, with nothing in the layout to stop it. Same class of check as the
    # mark-pitch warning in plot_probing.py.
    if n % 2 == 0:
        print(f"  ! {n} spokes puts a label at the bottom of the circle, where "
              f"the legend is -- they will overprint. Use an odd spoke count, "
              f"or move the legend.")

    # Explicit margins, not constrained_layout, because savefig.bbox is off (see
    # style.py): the saved page has to be exactly figsize or the include
    # rescales the text. Given in inches and converted, so the reserved space
    # stays fixed if the height ever changes.
    #
    #   top     0.32in  panel title, plus the "Object" spoke label below it
    #   bottom  0.34in  legend row
    #   sides   0.58in  the "LIBERO-10" / "All (mean)" spoke labels, which stick
    #                   out further than anything on the top or bottom, and are
    #                   the widest text on the figure
    #
    # The side reserve is generous on purpose. The two left spokes sit at 216
    # and 288 degrees, where the rim is curving *towards* the label, so a
    # radial pad that clears the arc at its tangent point still lets a corner
    # of the text cross it further along. Buying the clearance in the margin
    # rather than by shrinking the pad is what keeps the labels off the circle.
    # The height is chosen so the axes box comes out roughly square, i.e. no
    # taller than the circle the side margins already fix the width of. A taller
    # box does not enlarge the circle -- a polar axes inscribes it -- it just
    # adds white above and below, and the legend does not fill that white
    # because matplotlib anchors it to the circle's bounding box rather than to
    # the axes box.
    w = style.COL_WIDTH
    h = w * 0.87
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(projection="polar")
    fig.subplots_adjust(left=0.70 / w, right=1 - 0.70 / w,
                        bottom=0.47 / h, top=1 - 0.48 / h)

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
    # few points of clearance. Measured: this takes the worst spoke from touching
    # the rim to ~4pt clear.
    ax.set_xticklabels([])
    for ang, key in zip(angles, suites):
        # Screen direction of this spoke, given the offset/direction set above:
        # theta is measured clockwise from straight up.
        ux, uy = np.sin(ang), np.cos(ang)
        ax.annotate(
            SUITE_LABELS.get(key, key),
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
    # measures the title pad from the axes *box*, and the "Object" spoke label
    # is drawn outside that box -- so every pad small enough to look right
    # overprints the label, and every pad large enough to clear it leaves a
    # hole. A figure-coordinate anchor in the margin reserved above is
    # deterministic and does not depend on what the top spoke is called.
    fig.text(0.5, 1 - 0.155 / h, PANEL_TITLES.get(split, split),
             ha="center", va="center", fontweight="bold")

    # Legend below the axes, in a frame. Boxed rather than the frameless default
    # of style.py: it sits outside the axes over open page here, and the frame
    # is what stops it reading as a caption fragment. One row, three entries,
    # which is what the three-polygon limit above buys.
    #
    # The spacings are tight because three entries on one row is very close to
    # the page width at column size: at the matplotlib defaults this legend
    # measured wider than the figure and ran off both edges. Widen anything here
    # and re-check that the frame still fits inside figsize.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06),
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
    p.add_argument("csv", nargs="*", type=Path,
                   help="specific result CSVs (default: all "
                        "results/libero_radar_*.csv)")
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

    paths = args.csv or sorted(RESULTS_DIR.glob("libero_radar_*.csv"))
    if not paths:
        raise SystemExit(f"no libero_radar CSVs found in {RESULTS_DIR}")
    order = {s: i for i, s in enumerate(SPLIT_ORDER)}
    paths = sorted(paths, key=lambda q: (order.get(split_from(q), len(order)), q.stem))

    style.apply_style()

    # Local font-size override, applied after the shared style so it wins over
    # whichever module was swapped in above. See RADAR_PT.
    import matplotlib.pyplot as plt
    plt.rcParams.update({k: RADAR_PT for k in (
        "font.size", "axes.labelsize", "axes.titlesize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})

    dummy, missing = [], 0
    for path in paths:
        if not path.exists():
            raise SystemExit(f"no such results file: {path}")
        split = split_from(path)
        suites, values, methods, is_dummy = load(path)
        print_table(PANEL_TITLES.get(split, split), suites, values, methods)
        draw(split, suites, values, methods, path.stem + suffix)
        missing += sum(int(np.isnan(v).sum()) for v in values.values())
        if is_dummy:
            dummy.append(path.name)

    if missing:
        print(f"\n  note: {missing} empty cell(s) not yet run; the polygon is "
              f"broken at those spokes, not pulled to zero")
    if dummy:
        print("\n  " + "!" * 66)
        for n in dummy:
            print(f"  ! {n} is flagged DUMMY DATA -- these are placeholders,")
        print("  ! not measurements. Do not ship these figures in the paper.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
