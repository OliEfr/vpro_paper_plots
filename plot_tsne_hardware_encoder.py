r"""Real hardware, robot vs human video: does our learned tokenizer mix the embodiments?

Reads ``results/tsne_hardware_encoder.csv`` and ``results/tsne_hardware_encoder_metrics.csv``
(written by ``experiments/fit_tsne_hardware_encoder.py``) and emits

    figures/tsne_hardware_encoder[_science].pdf     t-SNE   (--embedding tsne, default)
    figures/umap_hardware_encoder[_science].pdf     UMAP    (--embedding umap)

Two DK1 dual-view LAM teachers, one config key apart: a frozen DINOv3-S/16 encoder (every
hardware LAM so far) against our learned patch tokenizer (the encoder used on LIBERO,
LIBERO-plus and MimicGen). Same dataset, same 30k budget, same geometry, same seed, same code
package. Columns are the two encoders, rows are the raw latent and a post-hoc control.

How to read it. Colour is the data source, so OVERLAP is the claim: if robot demos and human
video land on the same region, the latent is embodiment-invariant and a probe fitted on robot
actions can be expected to carry over to human video. Separated colours mean it cannot. The
number in each panel is source decodability (kNN-15, chance 0.5) measured on the latent
itself, not on the 2-D embedding -- read that number, not the apparent gap between blobs,
because t-SNE and UMAP preserve neighbourhoods rather than distances. Each panel's embedding
is fitted independently, so positions are comparable only within a panel.

The bottom row applies per-source standardisation at ANALYSIS time. It uses the source label,
so it is not evidence that either teacher learned an invariant latent; it shows that the two
sources occupy the same manifold offset in mean and scale, and how much of the separation is
that offset alone.

Both panels of a column share one action R^2 (episode-split ridge from the latent to the 7-D
end-effector action, robot rows). It is printed with the column title because embodiment
overlap is only interesting if the latent still knows about the action -- a collapsed latent
mixes the sources perfectly and is worth nothing.

Usage:
    python plot_tsne_hardware_encoder.py [--style paper|science] [--embedding tsne|umap]
                                         [--variants raw,srcstd]
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
VARIANT_LABELS = {"raw": "latent as trained",
                  "unitnorm": "unit-norm control",
                  "srcstd": "per-source standardised\n(post hoc)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--embedding", choices=["tsne", "umap"], default="tsne")
    p.add_argument("--variants", default="raw,srcstd",
                   help="comma-separated rows, from raw / unitnorm / srcstd")
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
    # One checkpoint per panel. Without this the metric lookup below matches every fitted
    # checkpoint at once and the in-panel numbers are ambiguous.
    ckpts = sorted(set(df.checkpoint.dropna()), key=lambda c: int(c) if str(c).isdigit() else 0)
    ck = a.checkpoint or (ckpts[-1] if ckpts else None)
    if ck is not None:
        df = df[df.checkpoint == ck]
        met = met[met.checkpoint == ck]
    met = met.set_index(["model", "variant"])
    variants = [v for v in a.variants.split(",") if v]
    models = [m for m in MODEL_ORDER if m in set(df.model)]
    assert models, f"no known model in {CSV.name}; found {sorted(set(df.model))}"
    colors = {"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]}
    fs = plt.rcParams["font.size"]
    x, y = f"{a.embedding}_x", f"{a.embedding}_y"

    w = style.TEXT_WIDTH
    fig, axes = plt.subplots(len(variants), len(models),
                             figsize=(w, w * 0.40 * len(variants) / len(models)), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.10, wspace=0.05, hspace=0.10)

    for r, variant in enumerate(variants):
        for c, model in enumerate(models):
            ax = axes[r][c]
            sub = df[(df.model == model) & (df.variant == variant)].sample(frac=1.0, random_state=0)
            assert len(sub), f"no rows for {model}/{variant}"
            ax.scatter(sub[x], sub[y], s=1.6, alpha=0.55, linewidths=0,
                       c=[colors[s] for s in sub.source], rasterized=True)
            ax.margins(0.06)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            m = met.loc[(model, variant)]
            # the metric belongs to the latent, not to the 2-D embedding: state it in-panel
            ax.text(0.03, 0.97, f"source kNN = {m.knn15_acc:.2f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=fs)
            if r == 0:
                ax.set_title(f"{m.model_label}\naction $R^2$ = {m.action_r2:.2f}",
                             fontsize=fs, pad=4)
            if c == 0:
                ax.set_ylabel(VARIANT_LABELS.get(variant, variant), fontsize=fs, labelpad=4)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[s],
                      markeredgecolor=style.MARKER_EDGE, markeredgewidth=0.4, markersize=4,
                      label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), handletextpad=0.3, columnspacing=1.6)
    suffix = "_science" if a.style == "science" else ""
    name = "tsne" if a.embedding == "tsne" else "umap"
    style.save(fig, f"{name}_hardware_encoder{suffix}")


if __name__ == "__main__":
    main()
