r"""Does the embodiment gap on real hardware open up during training, and does the encoder change that?

Reads ``results/tsne_hardware_encoder.csv`` and ``results/tsne_hardware_encoder_metrics.csv``
(written by ``experiments/fit_tsne_hardware_encoder.py``, which must have been given more than
one checkpoint per model) and emits

    figures/tsne_hardware_encoder_timeline[_science].pdf     t-SNE   (default)
    figures/umap_hardware_encoder_timeline[_science].pdf     UMAP    (--embedding umap)

The checkpoint twin of ``plot_tsne_hardware_encoder.py``: columns are the two encoders,
rows are training checkpoints, colour is the data source. It answers a question the
single-checkpoint figure cannot: whether robot demos and human video start mixed and separate
as training proceeds, or are separable from the start, and whether the learned tokenizer
follows a different trajectory from the frozen DINO encoder.

For the DINO baseline the answer is already known and is not encouraging: source decodability
runs 0.9754 at 5k to 0.9997 at 30k, so training makes the two sources MORE distinguishable,
not less. Whether our encoder does the same is the point of the comparison.

Read the number in each panel, not the gap between the blobs. It is source decodability
(kNN-15, chance 0.5) measured on the latent, while t-SNE and UMAP preserve neighbourhoods
rather than distances, and every panel is embedded independently.

Only the raw latent is shown, since the question is about the latent as trained. The
per-source-standardised control belongs to ``plot_tsne_hardware_encoder.py``.

Usage:
    python plot_tsne_hardware_encoder_timeline.py [--style paper|science]
                                                  [--embedding tsne|umap] [--variant raw]
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


def step_label(ckpt):
    """030000 -> '30k steps'. Keeps the panel labels short and in the paper's register."""
    try:
        n = int(ckpt)
    except (TypeError, ValueError):
        return str(ckpt)
    return f"{n // 1000}k steps" if n >= 1000 else f"{n} steps"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--embedding", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--variant", default="raw", help="which latent variant to draw (default raw)")
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
    assert "checkpoint" in df.columns, (
        f"{CSV.name} has no checkpoint column; refit with a checkpoint-aware "
        "fit_tsne_hardware_encoder.py")
    df = df[df.variant == a.variant]
    met = met[met.variant == a.variant].set_index(["model", "checkpoint"])

    models = [m for m in MODEL_ORDER if m in set(df.model)]
    ckpts = sorted(set(df.checkpoint.dropna()), key=lambda c: int(c) if str(c).isdigit() else 0)
    assert len(ckpts) > 1, (
        f"only one checkpoint present ({ckpts}); this figure needs at least two. Fit the 5k "
        "and 30k exports in the same run.")

    colors = {"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]}
    fs = plt.rcParams["font.size"]
    x, y = f"{a.embedding}_x", f"{a.embedding}_y"

    w = style.TEXT_WIDTH
    fig, axes = plt.subplots(len(ckpts), len(models),
                             figsize=(w, w * 0.40 * len(ckpts) / len(models)), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.10, wspace=0.05, hspace=0.10)

    for r, ck in enumerate(ckpts):
        for c, model in enumerate(models):
            ax = axes[r][c]
            sub = df[(df.model == model) & (df.checkpoint == ck)].sample(frac=1.0, random_state=0)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if not len(sub):   # a checkpoint may be missing for one arm; say so, do not fake it
                ax.text(0.5, 0.5, "not measured", transform=ax.transAxes, ha="center",
                        va="center", fontsize=fs, color=style.INK_MUTED)
                continue
            ax.scatter(sub[x], sub[y], s=1.6, alpha=0.55, linewidths=0,
                       c=[colors[s] for s in sub.source], rasterized=True)
            ax.margins(0.06)
            m = met.loc[(model, ck)]
            ax.text(0.03, 0.97, f"source kNN = {m.knn15_acc:.2f}\naction $R^2$ = {m.action_r2:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=fs)
            if r == 0:
                ax.set_title(m.model_label, fontsize=fs, pad=4)
            if c == 0:
                ax.set_ylabel(step_label(ck), fontsize=fs, labelpad=4)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[s],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4, markersize=4,
                      label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), handletextpad=0.3, columnspacing=1.6)
    suffix = "_science" if a.style == "science" else ""
    name = "tsne" if a.embedding == "tsne" else "umap"
    style.save(fig, f"{name}_hardware_encoder_timeline{suffix}")


if __name__ == "__main__":
    main()
