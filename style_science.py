"""Alternative figure style built on SciencePlots (garrettj403/SciencePlots).

Drop-in replacement for `style.py`: exposes the same names, so
`plot_probing.py --style science` swaps this in without any other change. Where
`style.py` reproduces the paper's exact Computer-Modern body text, this module
adopts the SciencePlots "science + ieee" look instead -- STIX serif, thin
boxed axes, ticks turned inward on all four sides, and the SciencePlots colour
set.

    pip install SciencePlots        # provides the 'science' / 'ieee' styles
    python plot_probing.py --style science

NO-LATEX NOTE. SciencePlots' 'science' style sets text.usetex=True and typesets
with a real LaTeX Computer Modern. This repo's build box has no LaTeX, so we add
the 'no-latex' style, which falls back to matplotlib's STIX serif. The layout,
sizes, ticks and colours are the SciencePlots ones; only the font substitutes.
If you build where LaTeX is installed and want the true SciencePlots text, drop
'no-latex' from STYLE_STACK below.

EMBEDDING IN LATEX -- identical to the paper style: built at \\textwidth, so
figure*, width=\\textwidth. See plot_probing.py's docstring for the snippet.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- registers the 'science'/'ieee' styles

STYLE_STACK = ["science", "ieee", "no-latex"]

# IEEEtran journal text block, same as style.py -- the figure is embedded at the
# same width regardless of which aesthetic draws it.
COL_WIDTH = 252 / 72.27
TEXT_WIDTH = 516 / 72.27

FIG_DIR = Path(__file__).resolve().parent / "figures"

# SciencePlots' own 'science' colour cycle: first three hues carry the suites,
# and their red is reused for the "ours" highlight -- so every colour on the
# figure is a SciencePlots colour, not a paste-in from the other style.
PALETTE = ["#0C5DA5", "#00B945", "#FF9500", "#FF2C00"]

# Marker channel is unchanged from the paper style: one shape per visual family
# so the four configs stay distinct, ordered plain -> bold onto "ours".
HATCHES = ["", "///", "...", "xxx"]
MARKERS = ["o", "^", "s", "X"]

# Contours: black by default, SciencePlots red on "ours".
MARKER_EDGE = "black"
MARKER_EDGE_OURS = "#FF2C00"

INK = "black"
INK_MUTED = "#474747"   # SciencePlots' grey, for the mean divider and shape keys
GRID = "#d0d0d0"


def apply_style():
    plt.style.use(STYLE_STACK)
    matplotlib.rcParams.update({
        # SciencePlots' 'science' sets savefig.bbox='tight', which crops the
        # canvas and defeats the exact-\textwidth sizing the include relies on.
        # Force it back off; the script reserves its own margins.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        # We draw a y-only grid in the script rather than the full SciencePlots
        # grid, so keep the style's grid off and colour ours here.
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        # Type 42 so IEEE PDF eXpress accepts the embedded fonts (rejects Type 3).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 200,
    })


def figsize(width="col", ratio=0.62):
    w = {"col": COL_WIDTH, "text": TEXT_WIDTH}.get(width, width)
    return (w, w * ratio)


def save(fig, name):
    """Write <name>.pdf (for LaTeX) and <name>.png (for quick viewing)."""
    FIG_DIR.mkdir(exist_ok=True)
    pdf, png = FIG_DIR / f"{name}.pdf", FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(FIG_DIR.parent)}")
    print(f"  wrote {png.relative_to(FIG_DIR.parent)}")
    return pdf
