r"""Latent-space embodiment mixing, start of training versus end.

Reads ``results/umap_teachers.csv`` and emits a grid over the three
embodiment-count teachers plus one figure per teacher:

    python plot_umap_teachers.py     # figures/umap_teachers.pdf + umap_teacher_{2,3,5}emb.pdf

WHAT IS PLOTTED. Each LAM teacher is trained on video from a different number of
robots (2, 3, or 5). Its 8-D latent is exported at several checkpoints, and one
UMAP is fitted per teacher jointly across all of them, so the panels of a single
teacher share coordinates and can be read as the same space at two moments.
Colour is the robot the frame came from.

The claim the figure has to carry is that embodiment stops being visible in the
latent as training proceeds -- the robots start in separable lobes and end
interleaved. That is a statement about two moments, so the figure shows two
columns, 1k and 70k, and not the full checkpoint sweep: the intermediate points
are in ``results/umap_decodability.csv`` as numbers, which is a better medium
for a monotone trend than five near-identical scatters.

COORDINATES ARE PER TEACHER, NEVER ACROSS. The three teachers are three separate
UMAP fits of three separately-trained latent spaces. Their axes have no common
meaning, so the axes carry no ticks -- a shared frame would invite exactly the
comparison that is invalid. What IS comparable across panels is the degree of
mixing within each one.

EACH PANEL IS CROPPED TO ITS OWN POINTS. The joint fit does put a teacher's two
checkpoints in one space, but training moves the manifold, so 1k and 70k occupy
different regions of it: drawn at the shared extent each checkpoint shrinks into
a corner and the mixing -- the thing being shown -- becomes unreadable. The
panels are therefore cropped individually. The cost is that POSITION between the
two columns carries no meaning; only the within-panel arrangement does. Nothing
in the claim rests on position, so this is a crop, not a distortion.

WHY NO MARKER SHAPES. style.py's palette rule asks for a second, non-colour
channel whenever hue alone separates series. It cannot be honoured here: the
marks are 1.5pt and a shape at that size is a blob. The palette was therefore
re-derived for five categories rather than reused blind -- see PALETTE below.

EMBEDDING IN LATEX
------------------
Built at 7.14in = \textwidth (43pc, IEEEtran journal), so the grid spans both
columns and needs figure*, not figure. The per-teacher figures are the same
width; use figure* for them too, or rebuild at style.COL_WIDTH.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/umap_teachers.pdf}
      \caption{...}
    \end{figure*}
"""
import argparse
from pathlib import Path

import pandas as pd

import style

DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "umap_teachers.csv"

# Robot -> colour, fixed by position in this list and shared with every other
# figure that colours by robot. Positions 0-3 are style.PALETTE unchanged; the
# fifth is Okabe-Ito black, chosen by re-deriving the floor rather than by eye.
#
# Worst-pair dE (CIE76, Vienot dichromat simulation) over ALL pairs:
#
#   style.PALETTE + black    normal 53.6   deuteranopia 16.4   protanopia 30.8
#   style.PALETTE            normal 53.6   deuteranopia 16.4   protanopia 30.8
#   style_science + black    normal 46.3   deuteranopia 16.0   protanopia 19.1
#   style_science            normal 46.3   deuteranopia 16.0   protanopia 19.1
#
# i.e. adding black costs nothing -- the binding pair is already inside the
# repo's own four, and black introduces no closer one. The three rejected
# candidates all did worse somewhere: sky collapses against blue in normal
# vision (26.4), vermillion against orange (33.4), yellow in protanopia (20.5).
EMB_ORDER = ["franka", "iiwa", "kinova3", "ur5e", "sawyer"]
# Filled in main(), AFTER the --style swap -- style_science carries a different
# PALETTE, and binding at import would silently render the science variant in
# the paper colours. Mutated in place so the helpers below see the update.
EMB_COLORS = {}

TEACHERS = [(2, "xemb1iiwa"), (3, "lib90t71"), (5, "xemb4all")]
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
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, plt, names, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=EMB_COLORS[n],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4,
                      markersize=4, label=n) for n in EMB_ORDER if n in names]
    fig.legend(handles=handles, ncol=len(handles), frameon=False,
               handletextpad=0.3, columnspacing=1.2, **kw)


def build_grid(df, name, plt):
    fig, axes = plt.subplots(2, 3, figsize=style.figsize("text", ratio=0.70))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.925, bottom=0.10,
                        wspace=0.10, hspace=0.10)
    for j, (n_emb, key) in enumerate(TEACHERS):
        sub_all = df[df["teacher"] == key]
        size = 1.1
        for i, (ckpt, _) in enumerate(COLS):
            panel(axes[i][j], sub_all[sub_all["checkpoint"] == ckpt], size)
        axes[0][j].set_title(f"{n_emb} embodiments", fontsize=_fs(), pad=4)
    for i, (_, label) in enumerate(COLS):
        axes[i][0].set_ylabel(label, fontsize=_fs())
    legend(fig, plt, EMB_ORDER, loc="lower center", bbox_to_anchor=(0.52, -0.012))
    style.save(fig, name)


def build_single(df, n_emb, key, name, plt):
    sub_all = df[df["teacher"] == key]
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("text", ratio=0.50))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.135, wspace=0.08)
    for ax, (ckpt, label) in zip(axes, COLS):
        panel(ax, sub_all[sub_all["checkpoint"] == ckpt], 1.6)
        ax.set_title(label, fontsize=_fs(), pad=4)
    legend(fig, plt, set(sub_all["embodiment"]), loc="lower center",
           bbox_to_anchor=(0.52, -0.015))
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
    build_grid(df, f"umap_teachers{suffix}", plt)
    for n_emb, key in TEACHERS:
        build_single(df, n_emb, key, f"umap_teacher_{n_emb}emb{suffix}", plt)


if __name__ == "__main__":
    main()
