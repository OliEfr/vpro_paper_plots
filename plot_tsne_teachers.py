r"""Latent-space embodiment mixing, start of training versus end -- t-SNE.

Reads ``results/tsne_teachers.csv`` and emits a grid over the three
embodiment-count teachers plus one figure per teacher:

    python plot_tsne_teachers.py     # figures/tsne_teachers.pdf + tsne_teacher_{2,3,5}emb.pdf

WHY THIS EXISTS ALONGSIDE plot_umap_teachers.py. The claim -- embodiment stops
being visible in the latent as training proceeds -- is a claim about the data,
not about UMAP. A referee is entitled to ask whether the neighbour-graph
embedder manufactured it. So the same latents, drawn from the same balanced
sample off the same seed, are also embedded with t-SNE; the two figures are
built to the same recipe and differ only in the projector. See
``experiments/fit_tsne_dumps.py``.

Everything below is inherited unchanged from the UMAP figure, and for the same
reasons: colour is the robot the frame came from; the panels are 1k and 70k
rather than the full sweep, because the monotone trend belongs in
``results/umap_decodability.csv`` as numbers; coordinates are per teacher and
never across, so the axes carry no ticks; each panel is cropped to its own
points, so position between columns carries no meaning and only the within-panel
arrangement does; and no marker shapes, because at 1.5pt a shape is a blob.

THE DECODABILITY NUMBERS ARE NOT REFITTED FOR t-SNE. They are computed on the
raw 8-D latent, never on a 2-D projection, so ``umap_decodability.csv`` is the
right table for this figure too despite the name. That is the point: the
quantitative claim never went through an embedder at all.

READING A t-SNE PANEL -- WHAT IS AND IS NOT INFORMATION. t-SNE preserves local
neighbourhoods. It does not preserve inter-cluster distance, and it does not
preserve cluster size: the gap between two blobs and the area of a blob are both
artefacts of the perplexity and the optimisation, not measurements. What is
readable is whether points of different colour are neighbours -- which is
exactly and only the claim this figure makes. Any caption should say so, or a
reader will measure the gaps.

EMBEDDING IN LATEX
------------------
Built at 7.14in = \textwidth (43pc, IEEEtran journal), so the grid spans both
columns and needs figure*, not figure. The per-teacher figures are the same
width; use figure* for them too, or rebuild at style.COL_WIDTH.

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/tsne_teachers.pdf}
      \caption{...}
    \end{figure*}
"""
import argparse
from pathlib import Path

import pandas as pd

import style

DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "tsne_teachers.csv"

# Robot -> colour, identical to plot_umap_teachers.py so the two projections of
# the same data are directly comparable panel for panel. Positions 0-3 are
# style.PALETTE unchanged; the fifth is Okabe-Ito black -- see the palette
# derivation in plot_umap_teachers.py, which this deliberately does not repeat.
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
    ax.scatter(sub["tsne_x"], sub["tsne_y"], s=size, alpha=0.55, linewidths=0,
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
    build_grid(df, f"tsne_teachers{suffix}", plt)
    for n_emb, key in TEACHERS:
        build_single(df, n_emb, key, f"tsne_teacher_{n_emb}emb{suffix}", plt)


if __name__ == "__main__":
    main()
