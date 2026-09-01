r"""Latent-space embodiment mixing at 2 and 5 embodiments, start versus end.

Reads ``results/umap_teachers.csv`` and emits one SINGLE-COLUMN 2x2 figure:

    python plot_umap_teachers_2x2.py   # figures/umap_teachers_2x2.pdf

              1k steps        70k steps
    2 emb   [ xemb1iiwa ]   [ xemb1iiwa ]
    5 emb   [ xemb4all  ]   [ xemb4all  ]

WHY THIS EXISTS ALONGSIDE plot_umap_teachers.py. That script builds the same
material at \textwidth: a 2x3 grid over all three teachers, for a figure*. This
one drops the 3-embodiment teacher and rebuilds the remaining four panels at
\columnwidth, for a plain figure. It is not a crop of the other -- see SIZE.

SIZE, AND WHY IT IS THE WHOLE POINT
-----------------------------------
Every figure in this repo is drawn with 8pt text (style.FIG_PT) and included at
1:1, so 8pt text lands on the page at 8pt. That contract only holds if the PDF
is authored at the width it will be included at. The \textwidth grid is 514pt
wide; dropped into a \columnwidth slot LaTeX rescales it by 251/514 = 0.488 and
its text arrives at 3.6pt -- half the size of every other single-column figure
in the paper, and unreadable in print. Measured, with `pdftotext -bbox`, median
word-box height when each figure is placed at \columnwidth:

    realworld_scaling.pdf            (col-width)   7.35pt
    realworld_multiview_vs_side.pdf  (col-width)   7.35pt
    umap_teachers.pdf                (text-width)  3.59pt   <- the problem
    umap_teachers_2x2.pdf            (col-width)   7.35pt   <- this figure

So this file is authored at style.COL_WIDTH and nothing rescales it.

(For reference, libero_radar_*.pdf sits at 9.19pt: those two radars carry a
documented local override to the 10pt body size, RADAR_PT in plot_libero_radar.
This figure follows the repo default instead, so it matches the majority.)

WHY THE AXES ARE BARE
---------------------
UMAP coordinates are not a measurement. The embedding has no units, no origin,
no meaningful scale, and its axes are not independent -- only the neighbourhood
structure survives the projection. Tick VALUES would therefore invite reading
distances off the page, which the method does not support, so there are none.
Axis NAMES ("UMAP 1"/"UMAP 2") would be honest but carry no information the
caption does not, and at \columnwidth across four panels they cost real drawing
area. Both are omitted; the caption says what the space is.

COORDINATES ARE PER TEACHER. The two rows are two separate UMAP fits of two
separately-trained latent spaces, so nothing about a position in the top row
relates to a position in the bottom one. Within a row the fit is joint across
checkpoints, but each panel is still cropped to its own points: training moves
the manifold, so at the shared extent each checkpoint shrinks into a corner and
the mixing -- the thing being shown -- stops being visible. What is comparable
between panels is the degree of mixing, never the location.

WHY NO MARKER SHAPES. style.py's palette rule asks for a second, non-colour
channel whenever hue alone separates series. It cannot be honoured here: the
marks are around 1pt and a shape at that size is a blob. The palette was
re-derived for five categories instead -- see EMB_ORDER.

EMBEDDING IN LATEX
------------------
    \begin{figure}[t]
      \centering
      \includegraphics[width=\columnwidth]{figures/umap_teachers_2x2.pdf}
      \caption{...}
    \end{figure}
"""
import argparse
from pathlib import Path

import pandas as pd

import style

DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "umap_teachers.csv"

# Robot -> colour, fixed by position and shared with plot_umap_teachers.py.
# Positions 0-3 are the style module's PALETTE unchanged; the fifth is Okabe-Ito
# black. Worst-pair dE (CIE76, Vienot dichromat simulation) over ALL pairs:
#
#   style.PALETTE + black    normal 53.6   deuteranopia 16.4   protanopia 30.8
#   style.PALETTE            normal 53.6   deuteranopia 16.4   protanopia 30.8
#   style_science + black    normal 46.3   deuteranopia 16.0   protanopia 19.1
#   style_science            normal 46.3   deuteranopia 16.0   protanopia 19.1
#
# i.e. adding black costs nothing: the binding pair is already inside the repo's
# own four and black introduces no closer one.
EMB_ORDER = ["franka", "iiwa", "kinova3", "ur5e", "sawyer"]
# Filled in main(), AFTER the --style swap -- style_science carries a different
# PALETTE, and binding at import would silently render the science variant in
# the paper colours. Mutated in place so the helpers below see the update.
EMB_COLORS = {}

# (teacher key, row label, marker area in pt^2). The two rows hold 4682 and
# 11705 points in panels of the same size, so a mark sized for the top row
# turns the bottom row into a solid block. Sized per row instead, which is the
# only way both rows show density rather than saturation.
ROWS = [("xemb1iiwa", "2 embodiments", 1.10),
        ("xemb4all",  "5 embodiments", 0.55)]
COLS = [("001000", "1k steps"), ("070000", "70k steps")]


def _fs():
    """Figure font size. Read from rcParams, not style.FIG_PT: that constant
    exists in style.py but not in style_science, so the science variant would
    crash on it. Same workaround as plot_libero_radar.py."""
    import matplotlib
    return matplotlib.rcParams["font.size"]


def panel(ax, sub, size, seed=0):
    # Draw in shuffled order. Plotting robot-by-robot puts whichever series is
    # drawn last on top of every overlap, which reads as that robot occupying
    # territory it merely covers -- the exact thing this figure measures.
    sub = sub.sample(frac=1.0, random_state=seed)
    ax.scatter(sub["umap_x"], sub["umap_y"], s=size, alpha=0.55, linewidths=0,
               c=[EMB_COLORS[e] for e in sub["embodiment"]], rasterized=True)
    # Per-panel autoscale with a small pad. NOT a shared limit across panels --
    # see the module docstring.
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def build(df, name, plt):
    # Explicit margins in inches, not constrained_layout: savefig.bbox is off
    # (see style.py), so the saved page must come out exactly figsize or the
    # include rescales it and the 8pt text stops being 8pt.
    #
    #   left    0.20in  rotated row label ("5 embodiments")
    #   top     0.17in  column titles
    #   bottom  0.30in  the 5-entry legend row
    #   right   0.02in  nothing to reserve, just keep the mark off the trim
    w = style.COL_WIDTH
    left_in, right_in, top_in, bot_in = 0.20, 0.02, 0.17, 0.30
    # Square panels: the axes box is (w - margins) / 2 wide, so two stacked rows
    # of the same height plus the reserved bands give the total height.
    panel_w = (w - left_in - right_in) / 2
    h = 2 * panel_w + top_in + bot_in

    fig, axes = plt.subplots(2, 2, figsize=(w, h))
    fig.subplots_adjust(left=left_in / w, right=1 - right_in / w,
                        top=1 - top_in / h, bottom=bot_in / h,
                        wspace=0.08, hspace=0.08)

    for i, (key, row_label, size) in enumerate(ROWS):
        sub_all = df[df["teacher"] == key]
        for j, (ckpt, _) in enumerate(COLS):
            panel(axes[i][j], sub_all[sub_all["checkpoint"] == ckpt], size)
        axes[i][0].set_ylabel(row_label, fontsize=_fs(), labelpad=2)
    for j, (_, col_label) in enumerate(COLS):
        axes[0][j].set_title(col_label, fontsize=_fs(), pad=3)

    # One legend for the whole figure, carrying all five robots even though the
    # top row only uses two: the colours are fixed across every figure in this
    # family, so a legend that shrank to the row above it would imply the
    # mapping had changed.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=EMB_COLORS[n],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4,
                      markersize=3.4, label=n) for n in EMB_ORDER]
    fig.legend(handles=handles, ncol=len(handles), frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.004),
               handletextpad=0.2, columnspacing=0.7, fontsize=_fs())
    style.save(fig, name)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    args = p.parse_args()

    global style
    if args.style == "science":
        import style_science
        style = style_science
    suffix = "_science" if args.style == "science" else ""

    style.apply_style()
    EMB_COLORS.update(zip(EMB_ORDER, list(style.PALETTE) + ["#000000"]))
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv, dtype={"checkpoint": str})
    build(df, f"umap_teachers_2x2{suffix}", plt)


if __name__ == "__main__":
    main()
