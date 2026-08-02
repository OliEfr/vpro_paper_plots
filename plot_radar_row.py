r"""All three radars in one row, as a two-column figure*.

Reads the three radar dumps and emits a single ``figures/radar_row.pdf``:

    python plot_radar_row.py

This is a THIRD view of data that already has two single-column figures each
(``libero_radar_{h,nonh}.pdf``, ``libero_plus_radar.pdf``). Those still build and
are still the ones to use if the three results are discussed apart. This file
exists for the case where they are discussed together, and it earns its place by
what it removes: three legends become one, three floats become one, and the
three polygons sit at the same scale so "the oracle envelope shrinks as the
setting gets harder" is a left-to-right reading rather than a page-flip.

PANEL ORDER is fixed left to right: non-held-out, held-out, LIBERO-Plus. That is
the order the results appear in the paper, and it runs easiest to hardest --
in-distribution tasks, then unseen tasks, then unseen tasks under disturbance.
Reordering PANELS changes the argument the figure makes, so it is a deliberate
edit rather than a cosmetic one.

CONFIG IS IMPORTED, NOT COPIED. Spoke orders, spoke labels, method labels,
titles and the palette all come from plot_libero_radar.py and
plot_libero_plus_radar.py by import. Those two are deliberately independent
copies of each other, but this file is a consumer of both: rename a suite or an
arm there and this figure follows, instead of disagreeing with the single-column
version of the same panel. Only the row LAYOUT lives here.

The three panels must agree on their arms, since one legend speaks for all of
them. main() checks that and refuses to build if they diverge -- a shared legend
over panels with different methods would silently mislabel two thirds of the
figure.

TEXT SIZE. 8pt (style.FIG_PT), NOT the 10pt the single-column radars use. Those
are set at body size because each fills a column on its own; here three panels
share \textwidth, so a panel is ~2.4in and 10pt labels would overwhelm the
circles they belong to. 8pt also matches probing.pdf, the repo's other figure*,
so the paper's two wide figures are set at the same size.

THE CIRCLES ARE SMALL, and that is the real cost of this figure. Three radars
across \textwidth leaves about 1.1in of diameter each once the spoke labels have
their gutters, because a label like "All (mean)" is over half the width of the
circle it hangs off. That is enough for the shape reading the radar is for, and
not enough to compare two nearby vertices by eye -- which the single-column
versions were already too coarse for. Exact values live in the printed table and
in the paper's tables.

EMBEDDING IN LATEX
------------------
Built at 7.140in = \textwidth (516pt, per IEEEtran journal mode), so this is a
figure*, NOT a plain figure, and it needs \usepackage{graphicx}. On IEEEtran a
figure* floats to the top of a page; \usepackage{stfloats} (already in the
preamble) lets it reach the bottom too.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/radar_row.pdf}
      \caption{Per-suite success rate breakdown at the 5\% action budget.
      \textbf{Left:} LIBERO tasks that receive action-labelled demonstrations.
      \textbf{Middle:} held-out LIBERO tasks, which never do (depth sweep).
      \textbf{Right:} LIBERO-Plus, broken out by disturbance dimension. Each
      spoke is one suite or one disturbance; \emph{All (mean)} aggregates the
      spokes to its left. The oracle latent stays near the rim throughout while
      the learned latent collapses inward as the setting gets harder.}
      \label{fig:radar_row}
    \end{figure*}

Include it at \textwidth and nothing is rescaled; any other width scales the
figure text off 8pt, which is the one thing the README forbids. The panels have
no (a)/(b)/(c) prefixes because they are one float with one caption -- the
caption says left/middle/right instead. If they ever need real subcaptions, that
is the point at which this should become three \subfloat'd panel PDFs rather
than one image, so \subcaption sets the letters and they cannot disagree with
the float they sit in.

Input schemas are the two documented in results/README.md; this script reads
both. Any column ending in ``_sr`` is an arm, in file order, and that order is
what fixes the colour, the dash and the marker -- so the three dumps must list
their arms weakest-first for a single legend to be true of all three panels.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

# The two single-column radar scripts, imported for their configuration only --
# see the docstring. Importing them does not run either one: both guard their
# main() behind __name__ == "__main__", and neither touches rcParams at import
# time, so their 10pt RADAR_PT override does not leak into this figure.
import plot_libero_radar as lib
import plot_libero_plus_radar as libp

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NAME = "radar_row"

# One entry per panel, left to right. Each names the dump, the column that
# indexes its spokes, the spoke order, the display names, the panel title, and
# where the radial scale column goes -- all pulled from whichever single-column
# script owns that panel, so this file holds no second copy of any of it.
PANELS = [
    dict(csv="libero_radar_nonh.csv", key="suite",
         order=lib.SUITE_ORDER, aggregate=lib.AGGREGATE,
         labels=lib.SUITE_LABELS, title=lib.PANEL_TITLES["nonh"],
         rlabel_deg=lib.RLABEL_DEG),
    dict(csv="libero_radar_h.csv", key="suite",
         order=lib.SUITE_ORDER, aggregate=lib.AGGREGATE,
         labels=lib.SUITE_LABELS, title=lib.PANEL_TITLES["h"],
         rlabel_deg=lib.RLABEL_DEG),
    dict(csv="libero_plus_radar.csv", key="dim",
         order=libp.DIM_ORDER, aggregate=libp.AGGREGATE,
         labels=libp.DIM_LABELS, title=libp.PANEL_TITLE,
         rlabel_deg=libp.RLABEL_DEG),
]

# Method display names. Both source scripts now carry the same three, which is
# the precondition for one shared legend; main() verifies it rather than
# assuming it.
METHOD_LABELS = dict(lib.METHOD_LABELS, **libp.METHOD_LABELS)

# Encoding channels, index-matched, straight from the single-column scripts so
# an arm looks the same here as it does there.
RADAR_PALETTE = lib.RADAR_PALETTE
DASHES = lib.DASHES
RADAR_MARKERS = lib.RADAR_MARKERS
FILL_ALPHA = lib.FILL_ALPHA

MEAN_ATOL = lib.MEAN_ATOL

# Layout reserves, in inches. Same approach as plot_libero_plus_radar.py: the
# margins are derived from named physical lengths rather than hard-coded
# fractions, so the circle diameter falls out of the width arithmetic instead of
# being guessed.
#
#   SIDE_IN         gutter outside the first and last panel, for the spoke
#                   labels that hang off them
#   GUTTER_IN       gutter BETWEEN two panels. Bigger than SIDE_IN, because it
#                   has to hold the right-hand label of one panel AND the
#                   left-hand label of the next without them touching.
#   TITLE_IN        panel title band at the very top
#   SPOKE_LABEL_IN  one 8pt line plus SPOKE_PAD_PT. Reserved above the circles
#                   for the top spoke label, and again below them when any panel
#                   has an even spoke count (LIBERO-Plus does: six spokes put a
#                   label at due south).
#   LEGEND_*        the shared legend row and its clearance
#
# SIDE_IN is sized off the widest spoke label in the figure, "All (mean)", which
# at 8pt Computer Modern is a little over half an inch. That is the number that
# decides how big the circles can be: every inch of gutter is an inch the three
# circles do not get.
#
# Both were measured off a render rather than guessed, and re-measuring is the
# way to change them: the widest label overhangs the circle's bounding box by
# ~0.53in on the left ("All (mean)") and ~0.37in on the right ("Camera"), so
# SIDE_IN = 0.53 + the page margin wanted, and GUTTER_IN = 0.90 + the clearance
# wanted between two adjacent panels' labels. A first pass at GUTTER_IN =
# 2*SIDE_IN left 0.36-0.41in of dead space between panels and cost every circle
# 0.14in of diameter.
SIDE_IN = 0.62      # -> ~0.09in of page margin past the widest left label
GUTTER_IN = 1.03    # -> ~0.13in between one panel's labels and the next's
TITLE_IN = 0.22
SPOKE_LABEL_IN = 0.155
LEGEND_GAP_IN = 0.10
# The framed legend measures ~0.20in tall at 8pt; the rest of this reserve is
# clearance between it and the south spoke label hanging below the circles.
LEGEND_ROW_IN = 0.28

# Clearance between the rim and the nearest edge of a spoke label, in points.
# 3.0 rather than the single-column figures' 4.0: the pad is an absolute length,
# so on a circle a third the diameter it reads proportionally larger.
SPOKE_PAD_PT = 3.0

# Marks and rules, scaled down from the single-column radars (1.2 / 3.6) for the
# smaller panels. Any smaller and the dash patterns stop being distinguishable,
# which is one of the three channels this figure's accessibility rests on.
LINE_WIDTH = 0.9
MARKER_SIZE = 2.4

RINGS = [25, 50, 75, 100]


def label_for(method):
    return METHOD_LABELS.get(method, method[:-3] if method.endswith("_sr") else method)


def load(panel):
    """Return (spoke keys, {method: values in %}, methods, is_dummy) for a panel.

    Generic over the two dump schemas: `key` names the index column (`suite` or
    `dim`) and `order` fixes the spokes. The aggregate, when the file carries
    one, is appended last and cross-checked against the mean over the spokes --
    same tolerance and same reasoning as the single-column scripts.
    """
    csv_path = RESULTS_DIR / panel["csv"]
    if not csv_path.exists():
        raise SystemExit(f"no such results file: {csv_path}")
    is_dummy = csv_path.read_text().lstrip().startswith("# DUMMY")

    df = pd.read_csv(csv_path, comment="#")
    methods = [c for c in df.columns if c.endswith("_sr")]
    if not methods:
        raise ValueError(f"{csv_path.name}: no columns ending in '_sr'")

    df = df.set_index(df[panel["key"]].astype(str).str.strip())
    missing = [k for k in panel["order"] if k not in df.index]
    if missing:
        raise SystemExit(f"{csv_path.name}: no row for {panel['key']}(s) "
                         f"{', '.join(missing)}")

    keys = list(panel["order"])
    values = {m: df.loc[keys, m].to_numpy(dtype=float) for m in methods}

    if panel["aggregate"] in df.index:
        keys.append(panel["aggregate"])
        for m in methods:
            dumped = float(df.loc[panel["aggregate"], m])
            recomputed = np.nanmean(values[m])
            if not np.isnan(recomputed) and not np.isclose(
                    dumped, recomputed, atol=MEAN_ATOL):
                print(f"  ! {csv_path.name}: {m} '{panel['aggregate']}' row is "
                      f"{dumped:.4f}, mean over spokes is {recomputed:.4f}")
            values[m] = np.append(values[m], dumped)

    for m in methods:
        if np.nanmax(values[m], initial=0.0) > 1.0:
            raise SystemExit(
                f"{csv_path.name}: {m} exceeds 1.0 -- success rates are stored "
                f"as fractions in [0, 1] and converted to percent by this script")
        values[m] = values[m] * 100.0

    return keys, values, methods, is_dummy


def print_table(panel, keys, values, methods):
    """Text table per panel -- identity is never carried by colour alone, and at
    this panel size it is the only place the numbers can be read at all."""
    width = max(12, max(len(panel["labels"].get(k, k)) for k in keys) + 1)
    head = f"{'spoke':<{width}}" + "".join(f"{label_for(m):>18}" for m in methods)
    print(f"\n  {panel['title']}  (success rate, %)")
    print("  " + head)
    print("  " + "-" * len(head))
    for i, k in enumerate(keys):
        if k == panel["aggregate"]:
            print("  " + "-" * len(head))
        cells = "".join(
            f"{values[m][i]:>18.4f}" if not np.isnan(values[m][i]) else f"{'--':>18}"
            for m in methods)
        print("  " + f"{panel['labels'].get(k, k):<{width}}" + cells)


def draw_panel(ax, panel, keys, values, methods):
    """Draw one radar into an existing polar axes. No legend, no title -- those
    are figure-level here, because they are shared or hand-placed."""
    import matplotlib.pyplot as plt

    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.append(angles, angles[0])

    ax.set_theta_offset(np.pi / 2)   # first spoke straight up
    ax.set_theta_direction(-1)       # then clockwise, as read

    for mi, m in enumerate(methods):
        v = np.append(values[m], values[m][0])
        colour = RADAR_PALETTE[mi % len(RADAR_PALETTE)]
        ax.plot(closed, v,
                linestyle=DASHES[mi % len(DASHES)], linewidth=LINE_WIDTH,
                marker=RADAR_MARKERS[mi % len(RADAR_MARKERS)],
                markersize=MARKER_SIZE, color=colour, markeredgecolor=colour,
                label=label_for(m), zorder=4 + mi)
        # fill(), not fill_between(): closed in theta, and a NaN spoke has to
        # leave a hole rather than being bridged. linestyle is pinned because
        # the SciencePlots prop cycle would otherwise hand the fill a dashed
        # pattern, which matplotlib scales by linewidth -- lw=0 makes it [0, 0]
        # and raises at draw time.
        ax.fill(closed, v, color=colour, alpha=FILL_ALPHA, linewidth=0,
                linestyle="solid", zorder=2 + mi)

    ax.set_ylim(0, 100)
    ax.set_yticks(RINGS)
    ax.set_yticklabels([str(r) for r in RINGS], color=style.INK_MUTED,
                       fontsize=plt.rcParams["font.size"] * 0.8)
    ax.set_rlabel_position(panel["rlabel_deg"])

    ax.set_xticks(angles)
    # Placed by hand for the reason given at length in the single-column
    # scripts: matplotlib pads a polar tick label radially but anchors its box
    # on the arc, so on an angled spoke a corner crosses the rim whatever the
    # pad. Anchoring on the side facing the circle makes the box grow outward.
    ax.set_xticklabels([])
    for ang, key in zip(angles, keys):
        ux, uy = np.sin(ang), np.cos(ang)
        ax.annotate(
            panel["labels"].get(key, key),
            xy=(ang, 100), xycoords="data",
            xytext=(SPOKE_PAD_PT * ux, SPOKE_PAD_PT * uy),
            textcoords="offset points",
            ha="center" if abs(ux) < 0.35 else ("left" if ux > 0 else "right"),
            va="center" if abs(uy) < 0.35 else ("bottom" if uy > 0 else "top"),
            annotation_clip=False,
        )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    ax.grid(color=style.GRID, linewidth=0.4, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color(style.INK_MUTED)
    ax.spines["polar"].set_linewidth(0.7)
    # Solid, set explicitly -- the rim inherits a dashed pattern under the
    # SciencePlots style, which blurs the 100% boundary and raises at draw time.
    ax.spines["polar"].set_linestyle("solid")
    ax.set_facecolor("white")


def draw(panels, loaded, name):
    """Lay the three panels out in a row and write figures/<name>.pdf."""
    import matplotlib.pyplot as plt

    w = style.TEXT_WIDTH
    npanel = len(panels)

    # The circle diameter is what is left of the width after the gutters, not a
    # number chosen up front -- see SIDE_IN. Each panel's axes rect IS the
    # circle's bounding box; the spoke labels live in the gutters around it.
    diam = (w - 2 * SIDE_IN - (npanel - 1) * GUTTER_IN) / npanel
    if diam <= 0:
        raise SystemExit("gutters exceed the page width; reduce SIDE_IN")

    # A panel with an even spoke count puts a label at due south, between the
    # circles and the shared legend. LIBERO-Plus has six spokes, so the band is
    # normally reserved; it collapses to zero if every panel is odd.
    band = SPOKE_LABEL_IN if any(len(k) % 2 == 0 for k, _, _, _ in loaded) else 0.0

    bottom_in = LEGEND_ROW_IN + LEGEND_GAP_IN + band
    top_in = TITLE_IN + SPOKE_LABEL_IN
    h = bottom_in + diam + top_in

    # Explicit geometry, not constrained_layout, because savefig.bbox is off
    # (see style.py): the saved page has to be exactly figsize or
    # \includegraphics[width=\textwidth] rescales it and the 8pt text stops
    # being 8pt on the page.
    fig = plt.figure(figsize=(w, h))
    axes = []
    for i, (panel, (keys, values, methods, _)) in enumerate(zip(panels, loaded)):
        x0 = SIDE_IN + i * (diam + GUTTER_IN)
        ax = fig.add_axes([x0 / w, bottom_in / h, diam / w, diam / h],
                          projection="polar")
        draw_panel(ax, panel, keys, values, methods)
        axes.append(ax)
        # Title centred over the circle, in the band reserved at the top. Placed
        # in figure coordinates rather than with set_title() so its distance
        # from the page edge does not depend on how tall the top spoke label is.
        fig.text((x0 + diam / 2) / w, 1 - (TITLE_IN / 2) / h, panel["title"],
                 ha="center", va="center", fontweight="bold")

    # ONE legend for all three panels -- the whole point of the row. Taken from
    # the first panel's handles, which main() has already checked are the same
    # arms in the same order as the other two.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.5, (LEGEND_GAP_IN / 2) / h),
                     ncol=len(labels), frameon=True, fancybox=False,
                     edgecolor=style.INK_MUTED, framealpha=1.0,
                     borderaxespad=0, handlelength=1.6, handletextpad=0.4,
                     columnspacing=1.4, borderpad=0.4)
    leg.get_frame().set_linewidth(0.6)

    return style.save(fig, name)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["paper", "science"], default="paper",
                   help="paper: exact IEEEtran/Computer-Modern match (default). "
                        "science: the SciencePlots aesthetic (garrettj403/SciencePlots).")
    args = p.parse_args()

    global style
    suffix = ""
    if args.style == "science":
        import style_science
        style = style_science
        suffix = "_science"

    loaded = [load(panel) for panel in PANELS]

    # One legend speaks for all three panels, so all three must carry the same
    # arms in the same order. Column order is what fixes colour, dash and
    # marker, so a divergence here would not just mislabel the legend -- it
    # would put two different policies under one entry.
    reference = [label_for(m) for m in loaded[0][2]]
    for panel, (_, _, methods, _) in zip(PANELS[1:], loaded[1:]):
        got = [label_for(m) for m in methods]
        if got != reference:
            raise SystemExit(
                f"{panel['csv']}: arms {got} do not match {PANELS[0]['csv']}'s "
                f"{reference}. One shared legend cannot describe both -- align "
                f"the dumps' _sr columns, or build these as separate figures.")

    style.apply_style()

    for panel, (keys, values, methods, _) in zip(PANELS, loaded):
        print_table(panel, keys, values, methods)
    draw(PANELS, loaded, NAME + suffix)

    missing = sum(int(np.isnan(v).sum())
                  for _, values, _, _ in loaded for v in values.values())
    if missing:
        print(f"\n  note: {missing} empty cell(s) not yet run; the polygon is "
              f"broken at those spokes, not pulled to zero")

    dummy = [panel["csv"] for panel, (_, _, _, d) in zip(PANELS, loaded) if d]
    if dummy:
        print("\n  " + "!" * 66)
        for n in dummy:
            print(f"  ! {n} is flagged DUMMY DATA -- these are placeholders,")
        print("  ! not measurements. Do not ship this figure in the paper.")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
