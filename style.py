"""Shared matplotlib style for IEEEtran two-column paper figures.

Import `apply_style()` once at the top of a plot script, size figures with
`figsize()`, and write them out with `save()`. Nothing here is specific to a
particular plot -- new plot scripts should reuse it rather than re-deriving
figure widths and font sizes.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# IEEEtran text block, in inches. A figure rendered at exactly these widths and
# included with [width=\columnwidth] / [width=\textwidth] is not rescaled by
# LaTeX, so the font sizes below survive into the PDF at their stated point size.
COL_WIDTH = 3.5   # \columnwidth  (single column)
TEXT_WIDTH = 7.16  # \textwidth    (spans both columns, use with figure*)

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
HATCHES = ["", "///", "...", "xxx"]
MARKERS = ["o", "s", "^", "D"]

INK = "#1a1a1a"       # primary text
INK_MUTED = "#6b6b6b"  # axis labels, ticks
GRID = "#d9d9d9"


def apply_style():
    """Set rcParams to match IEEEtran body text."""
    matplotlib.rcParams.update({
        # IEEEtran sets Times; a serif figure font keeps captions and axis
        # labels from looking pasted in from another document.
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # Body text is 10pt. Figure text one to three points smaller reads as
        # subordinate without becoming illegible at print size.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Recessive axes and grid: the data should be the darkest thing here.
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,

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
