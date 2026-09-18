r"""Real-hardware latent, robot vs human, in three views of the SAME latent -- t-SNE.

Reads ``results/tsne_hardware_motion.csv`` (+ ``_decodability.csv``), written by
``experiments/fit_tsne_hardware_motion.py``, and emits
``figures/tsne_hardware_motion[_science].pdf``.

Question answered: is the robot/human separation of the hardware latent
(plot_tsne_hardware.py) carried by the ACTION-relevant part of the latent, or by
pose/appearance content the policy does not need? Per LAM, three projections of
the identical 30k-step latents are embedded separately:

    raw latent (8-D)            what plot_tsne_hardware.py shows
    motion (xyz) subspace       the 3 latent directions a ridge probe uses to predict the
                                realized EE motion state[t+5]-state[t] (fitted on robot rows)
    minus top-3 pose directions the latent with the 3 directions most predictive of the
                                ABSOLUTE EE pose removed

The panel titles carry the kNN-15 source decodability fitted on the projected
latent itself (chance 0.5). t-SNE preserves neighbourhoods, not distances: read
the colour mixing and the kNN number, never the gap.

Usage:
    python plot_tsne_hardware_motion.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "tsne_hardware_motion.csv"
DEC = HERE / "results" / "tsne_hardware_motion_decodability.csv"
SOURCE_ORDER = ["robot_3cam", "video_2cam"]
SOURCE_LABELS = {"robot_3cam": "robot demos", "video_2cam": "human video"}
MODELS = [("sharedlam2", "multi-view LAM (front + side)"), ("sharedlam3side", "single-view LAM (side)")]
VIEWS = [("raw latent (8-D)", "raw latent (8-D)"),
         ("motion (xyz) subspace (3-D)", "motion-predictive subspace (3-D)"),
         ("minus top-3 pose directions (5-D)", "pose directions removed (5-D)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]}
    df = pd.read_csv(CSV)
    dec = pd.read_csv(DEC).set_index(["model", "view"]).knn15
    fs = plt.rcParams["font.size"]

    fig, axes = plt.subplots(len(MODELS), len(VIEWS), figsize=style.figsize("text", ratio=0.62))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.90, bottom=0.08, wspace=0.08, hspace=0.22)
    for i, (mkey, mlabel) in enumerate(MODELS):
        for j, (vkey, vlabel) in enumerate(VIEWS):
            ax = axes[i][j]
            sub = df[(df.model == mkey) & (df.view == vkey)].sample(frac=1.0, random_state=0)
            ax.scatter(sub.tsne_x, sub.tsne_y, s=1.2, alpha=0.55, linewidths=0,
                       c=[colors[s] for s in sub.source], rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.margins(0.06)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{vlabel}\nsource kNN = {dec[(mkey, vkey)]:.2f}" if i == 0
                         else f"source kNN = {dec[(mkey, vkey)]:.2f}", fontsize=fs, pad=3)
        axes[i][0].set_ylabel(mlabel, fontsize=fs)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[s],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4, markersize=4,
                      label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.52, -0.005),
               handletextpad=0.3, columnspacing=1.6)
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"tsne_hardware_motion{suffix}")


if __name__ == "__main__":
    main()
