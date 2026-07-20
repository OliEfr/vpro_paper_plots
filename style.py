"""Shared matplotlib style for IEEEtran two-column paper figures.

Import `apply_style()` once at the top of a plot script, size figures with
`figsize()`, and write them out with `save()`. Nothing here is specific to a
particular plot -- new plot scripts should reuse it rather than re-deriving
figure widths and font sizes.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# IEEEtran text block, in inches, read out of IEEEtran.cls for the paper's
# actual \documentclass[10pt,journal] -- not the conference or compsoc values,
# which differ. From the journal branch of the class:
#
#   \textwidth  43pc = 516 TeX pt      \columnsep 1pc = 12 TeX pt
#   \columnwidth = (516 - 12) / 2 = 252 TeX pt
#
# TeX pt is 1/72.27 in, not the 1/72 in of a PostScript point, hence the odd
# decimals. A figure rendered at exactly these widths and included with
# [width=\columnwidth] / [width=\textwidth] is not rescaled by LaTeX, so the
# font sizes below land on the page at their stated point size.
COL_WIDTH = 252 / 72.27   # 3.487in  \columnwidth  (single column)
TEXT_WIDTH = 516 / 72.27  # 7.140in  \textwidth    (both columns, use figure*)

# Figure text size, in points. 8pt = IEEEtran's \footnotesize, the size the
# class already uses for captions -- so figure labels sit a clear step below the
# 10pt body, which reads as subordinate (the conventional choice for figures).
# The document body is 10pt; set FIG_PT_TEX = 10 to match it instead.
#
# The conversion is not decorative: a size in TeX points (1/72.27in) is not the
# same as a matplotlib font size, which is in PostScript points (1/72in). Passing
# a bare 8 to matplotlib would render 0.37% large. Invisible in practice, but the
# point of this file is that the number is exactly right, not approximately.
FIG_PT_TEX = 8
FIG_PT = FIG_PT_TEX * 72 / 72.27  # 7.970 PostScript pt

FIG_DIR = Path(__file__).resolve().parent / "figures"

# Categorical palette, fixed order -- assign by position, never cycle or recolor.
# Okabe-Ito subset. Checked over ALL pairs, not just adjacent ones: in a
# grouped scatter every series sits next to every other, so the weaker
# adjacent-only check would not have covered it.
#
#   normal vision   worst pair dE 18.7  (green/blue)      -- clear
#   deuteranopia    worst pair dE  7.6  (green/pink)      -- floor band
#
# That 7.6 is in the 6-8 band, which is permissible ONLY while a second,
# non-color channel also distinguishes the series. That channel is HATCHES /
# MARKERS below. If you ever drop the marker shapes and let hue carry identity
# alone, this palette stops being accessible -- re-derive it first.
PALETTE = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]

# Secondary encoding, same fixed order. Two jobs: the CVD floor above, and the
# fact that IEEE proceedings still get printed and photocopied in grayscale.
# HATCHES for filled marks (bars), MARKERS for point marks.
#
# The marker set is chosen for maximum mutual distinctness at small print size:
# one shape from each visual family -- round (o), pointed (^), blocky (s), spiky
# (X) -- so no two read as the same blob the way circle/square/diamond did.
# Thin "+"/"x" were rejected: they thin out and vanish below ~4pt in print.
# Order also runs plain -> bold, so "ours" (4th) gets the most salient mark.
HATCHES = ["", "///", "...", "xxx"]
MARKERS = ["o", "^", "s", "X"]

# Marker contours. Every mark gets a near-black edge for definition against the
# light background (it also rescues the lighter palette fills, which sit under
# 3:1 contrast on their own). The "ours" method's mark gets a red edge instead,
# so the hero configuration is flagged in every suite colour at once. Red is a
# highlight here, never a data series -- it only ever appears as this edge.
MARKER_EDGE = "#1a1a1a"
MARKER_EDGE_OURS = "#E31A1C"

INK = "#1a1a1a"       # primary text
INK_MUTED = "#6b6b6b"  # axis labels, ticks
GRID = "#d9d9d9"


def apply_style():
    """Set rcParams: Computer Modern at IEEEtran footnotesize (see FIG_PT)."""
    matplotlib.rcParams.update({
        # main.tex loads no font package (no times/newtx/mathptmx/lmodern), so
        # the document renders in Computer Modern. matplotlib ships CM, so the
        # figure can use the same typeface rather than an approximation.
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # cmr10 has no glyph for the Unicode minus; route numeric text through
        # mathtext so a negative tick label cannot render as a missing box.
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,

        # Every piece of figure text is set at FIG_PT (8pt footnotesize), so a
        # label lands on the page at that exact size -- but only if the figure is
        # included at 1:1, see figsize().
        "font.size": FIG_PT,
        "axes.labelsize": FIG_PT,
        "axes.titlesize": FIG_PT,
        "xtick.labelsize": FIG_PT,
        "ytick.labelsize": FIG_PT,
        "legend.fontsize": FIG_PT,

        # All text is black. The spines, tick marks and grid stay recessive
        # grey (they are rules, not text), but every label -- axis titles and
        # tick labels alike -- is pure black. labelcolor is set separately from
        # the tick `color` so the tick *marks* can stay grey while their labels
        # go black.
        "text.color": "black",
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": "black",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handleheight": 0.9,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "hatch.linewidth": 0.5,

        "figure.dpi": 200,
        # NOT bbox="tight". Tight cropping shrinks the saved canvas to whatever
        # the content happens to occupy, so the PDF comes out a fraction wider
        # or narrower than figsize; \includegraphics[width=\textwidth] then
        # rescales it, and the 10pt text above lands on the page at something
        # other than 10pt. Scripts reserve their own margins with
        # subplots_adjust instead, and the saved page is exactly figsize.
        "savefig.bbox": None,

        # Type 42 embeds TrueType outlines rather than Type 3 bitmapped fonts.
        # IEEE PDF eXpress rejects Type 3.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figsize(width="col", ratio=0.62):
    """Figure size in inches.

    width: "col" for \\columnwidth, "text" for \\textwidth, or a number.
    ratio: height / width. 0.62 is roughly golden and leaves room for a caption.
    """
    w = {"col": COL_WIDTH, "text": TEXT_WIDTH}.get(width, width)
    return (w, w * ratio)


def save(fig, name):
    """Write <name>.pdf (for LaTeX) and <name>.png (for quick viewing).

    The PDF is the one to include in the paper -- it stays vector, so it is
    still sharp when the reviewer zooms in.
    """
    FIG_DIR.mkdir(exist_ok=True)
    pdf, png = FIG_DIR / f"{name}.pdf", FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(FIG_DIR.parent)}")
    print(f"  wrote {png.relative_to(FIG_DIR.parent)}")
    return pdf
