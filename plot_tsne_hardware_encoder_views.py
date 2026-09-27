r"""Where the robot-vs-human split lives in the latent, for both encoders.

Reads ``results/tsne_hardware_encoder.csv`` and ``results/tsne_hardware_encoder_metrics.csv``
(written by ``experiments/fit_tsne_hardware_encoder.py``) and emits

    figures/tsne_hardware_encoder_views[_science].pdf     t-SNE  (default)
    figures/umap_hardware_encoder_views[_science].pdf     UMAP   (--embedding umap)

The encoder-comparison counterpart of ``plot_tsne_hardware_motion.py``: columns are the two
encoders, rows are three views of the SAME 30k latent, colour is the data source.

  raw latent (8-D)                everything the LAM encodes
  motion subspace (3-D)           the part that predicts end-effector translation, i.e. the
                                  top-3 directions of a ridge map from the latent to
                                  state[t+5]-state[t], fitted on robot rows
  minus top-3 pose directions     the latent with absolute arm configuration projected out

The point of the decomposition: if robot and human video separate in the raw latent but mix
once only the action-relevant part is kept, the gap is appearance and pose rather than motion,
and a probe trained on robot actions may still transfer. If they stay separated even in the
motion subspace, the gap is in the part that matters.

Both subspaces are fitted on robot rows only, on all valid rows before sampling, exactly as
the existing hardware motion figure does.

One number here is NOT comparable to the published motion figure. This script measures source
decodability under an EPISODE-grouped split, while ``fit_tsne_hardware_motion.py`` used a
plain 5-fold split that lets frames of one episode fall on both sides and so reads high.

Usage:
    python plot_tsne_hardware_encoder_views.py [--style paper|science]
                                               [--embedding tsne|umap] [--checkpoint 030000]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "tsne_hardware_encoder.csv"
METRICS = HERE / "results" / "tsne_hardware_encoder_metrics.csv"

SOURCE_ORDER = ["robot_3cam", "video_2cam"]
SOURCE_LABELS = {"robot_3cam": "robot demos", "video_2cam": "human video"}
MODEL_ORDER = ["ours_dino", "ours_learned"]
VIEWS = [("raw", "raw latent (8-D)"),
         ("motion", "motion subspace (3-D)"),
         ("pose_removed", "pose directions\nremoved (8-D)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--embedding", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--checkpoint", default=None,
                   help="which checkpoint to draw (default: the last one present)")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(CSV, dtype={"checkpoint": str})
    met = pd.read_csv(METRICS, dtype={"checkpoint": str})
    have = set(df.variant)
    views = [(k, lab) for k, lab in VIEWS if k in have]
    assert len(views) == len(VIEWS), (
        f"missing latent spaces {sorted({k for k, _ in VIEWS} - have)}; refit with a "
        "fit_tsne_hardware_encoder.py that computes the motion / pose subspaces")

    ckpts = sorted(set(df.checkpoint.dropna()), key=lambda c: int(c) if str(c).isdigit() else 0)
    ck = a.checkpoint or (ckpts[-1] if ckpts else None)
    df = df[df.checkpoint == ck]
    met = met[met.checkpoint == ck].set_index(["model", "variant"])
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    assert models, f"no known model at checkpoint {ck}; found {sorted(set(df.model))}"

    colors = {"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]}
    fs = plt.rcParams["font.size"]
    x, y = f"{a.embedding}_x", f"{a.embedding}_y"

    w = style.TEXT_WIDTH
    fig, axes = plt.subplots(len(views), len(models),
                             figsize=(w, w * 0.40 * len(views) / len(models)), squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.92, bottom=0.085, wspace=0.05, hspace=0.10)

    for r, (vkey, vlabel) in enumerate(views):
        for c, model in enumerate(models):
            ax = axes[r][c]
            sub = df[(df.model == model) & (df.variant == vkey)].sample(frac=1.0, random_state=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if not len(sub):
                ax.text(0.5, 0.5, "not measured", transform=ax.transAxes, ha="center",
                        va="center", fontsize=fs, color=style.INK_MUTED)
                continue
            ax.scatter(sub[x], sub[y], s=1.6, alpha=0.55, linewidths=0,
                       c=[colors[s] for s in sub.source], rasterized=True)
            ax.margins(0.06)
            m = met.loc[(model, vkey)]
            ax.text(0.03, 0.97, f"source kNN = {m.knn15_acc:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=fs)
            if r == 0:
                ax.set_title(f"{m.model_label}\naction $R^2$ = {m.action_r2:.2f}",
                             fontsize=fs, pad=4)
            if c == 0:
                ax.set_ylabel(vlabel, fontsize=fs, labelpad=4)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[s],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4, markersize=4,
                      label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.012), handletextpad=0.3, columnspacing=1.6)
    suffix = "_science" if a.style == "science" else ""
    name = "tsne" if a.embedding == "tsne" else "umap"
    style.save(fig, f"{name}_hardware_encoder_views{suffix}")


if __name__ == "__main__":
    main()
