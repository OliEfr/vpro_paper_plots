r"""Multi-view hardware LAM, robot vs human, first vs last checkpoint -- t-SNE, 1x2.

Reads ``results/tsne_hardware_motion_timeline.csv`` (+ ``_decodability.csv``), written
by ``experiments/fit_tsne_hardware_motion_timeline.py``, and emits one figure per view:

    figures/tsne_hardware_timeline_raw[_science].pdf           the 8-D latent
    figures/tsne_hardware_timeline_motion[_science].pdf        motion-predictive subspace (3-D)
    figures/tsne_hardware_timeline_pose_removed[_science].pdf  top-3 pose directions removed (5-D)

The hardware analogue of plot_tsne_teachers_2x2.py's "1k steps vs 70k steps" reading,
reduced to one LAM (front+side) and one row: left panel 1k steps, right panel 30k
steps, colour = data source. Both panels embed the identical 55-episode frame subset.
Panel titles carry the kNN-15 source decodability on the projected latent (chance
0.5). t-SNE keeps neighbourhoods, not distances: read colour mixing and the kNN
number, never the gap between blobs.

Usage:
    python plot_tsne_hardware_motion_timeline.py [--style paper|science] [--view raw|motion|pose_removed|all]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "tsne_hardware_motion_timeline.csv"
DEC = HERE / "results" / "tsne_hardware_motion_timeline_decodability.csv"
SOURCE_ORDER = ["robot_3cam", "video_2cam"]
SOURCE_LABELS = {"robot_3cam": "robot demos", "video_2cam": "human video"}
CKPTS = [("001000", "1k steps"), ("030000", "30k steps")]
VIEW_TITLES = {"raw": "raw latent (8-D)", "motion": "motion-predictive subspace (3-D)",
               "pose_removed": "pose directions removed (5-D)"}


def build(df, dec, view, style, plt, suffix):
    from matplotlib.lines import Line2D
    colors = {"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]}
    fs = plt.rcParams["font.size"]
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("col", ratio=0.62))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14, wspace=0.08)
    for ax, (ck, label) in zip(axes, CKPTS):
        sub = df[(df.view == view) & (df.checkpoint == ck)].sample(frac=1.0, random_state=0)
        ax.scatter(sub.tsne_x, sub.tsne_y, s=1.6, alpha=0.55, linewidths=0,
                   c=[colors[s] for s in sub.source], rasterized=True)
        ax.margins(0.06)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{label}   (source kNN = {dec[(ck, view)]:.2f})", fontsize=fs, pad=3)
    fig.suptitle(VIEW_TITLES[view], fontsize=fs, y=0.99)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[s],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4, markersize=4,
                      label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               handletextpad=0.3, columnspacing=1.6)
    style.save(fig, f"tsne_hardware_timeline_{view}{suffix}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--view", choices=list(VIEW_TITLES) + ["all"], default="all")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    df = pd.read_csv(CSV, dtype={"checkpoint": str})
    dec = pd.read_csv(DEC, dtype={"checkpoint": str}).set_index(["checkpoint", "view"]).knn15
    suffix = "_science" if a.style == "science" else ""
    views = list(VIEW_TITLES) if a.view == "all" else [a.view]
    for v in views:
        build(df, dec, v, style, plt, suffix)
        plt.close("all")


if __name__ == "__main__":
    main()
