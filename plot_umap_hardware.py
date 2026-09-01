r"""Real-hardware latent: robot demonstrations versus human video, over training.

Reads ``results/umap_hardware.csv`` and emits:

    python plot_umap_hardware.py   # figures/umap_hardware.pdf
                                   # figures/umap_hardware_sharedlam2.pdf

THE POINT OF THIS FIGURE IS THAT IT DOES NOT LOOK LIKE THE SIMULATION ONE. In
LIBERO the robots start separable and end interleaved (plot_umap_teachers.py).
On real hardware the two data sources -- teleoperated robot episodes and
human-style video -- are separable at the FIRST checkpoint and stay separable at
the last. Source decodability sits at kNN 0.98-1.00 from 5k onward for all three
LAMs, and never falls (results/umap_decodability.csv). Training does not align
them; it sharpens them.

That contrast is the reason both figures exist, so they are deliberately built
to the same recipe -- same UMAP settings, same balanced sampling, same two-column
crop -- and differ only in what colour encodes. A reader putting them side by
side should be able to attribute the difference to the data and not to the
plotting.

WHICH EARLY CHECKPOINT. Only sharedlam2 has a genuine 1k export, inherited from
the 3k screening run that shares its config, seed, data and LR schedule. The
other two LAMs were saved every 5k from 5k, so 5k is the earliest that exists
for them -- no 1k checkpoint was ever written, and producing one means retraining.
The grid therefore uses 5k for all three, which is a matched comparison at a
slightly later point; ``umap_hardware_sharedlam2`` uses the real 1k and is the
closer analogue of the 70k simulation figure. Both are honest; they answer
slightly different questions, and the caption should say which one it is.

sharedlam4tl is trained on a different dataset (the new-task set: 531 robot /
1000 video episodes against 681 / 1040 for the other two), so its panel is not
strictly comparable to its neighbours even at a matched step.

EMBEDDING IN LATEX
------------------
Built at 7.14in = \textwidth (43pc, IEEEtran journal); both need figure*.
"""
import argparse
from pathlib import Path

import pandas as pd

import style

DEFAULT_CSV = Path(__file__).resolve().parent / "results" / "umap_hardware.csv"

# Two sources, so two colours from the repo palette rather than a re-derivation:
# positions 0 and 2 (blue / orange), the pair with the widest separation under
# both dichromat simulations of the four.
SOURCE_ORDER = ["robot_3cam", "video_2cam"]
SOURCE_COLORS = {}  # filled in main(), after the --style swap (see plot_umap_teachers.py)
SOURCE_LABELS = {"robot_3cam": "robot demos", "video_2cam": "human video"}

MODELS = [("sharedlam2", "front + side"),
          ("sharedlam3side", "side only"),
          ("sharedlam4tl", "front + side, new tasks")]


def _fs():
    import matplotlib
    return matplotlib.rcParams["font.size"]


def panel(ax, sub, size, seed=0):
    # Shuffle so neither source is systematically drawn on top of the other.
    sub = sub.sample(frac=1.0, random_state=seed)
    ax.scatter(sub["umap_x"], sub["umap_y"], s=size, alpha=0.55, linewidths=0,
               c=[SOURCE_COLORS[s] for s in sub["source"]], rasterized=True)
    ax.margins(0.06)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def legend(fig, plt, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=SOURCE_COLORS[s], markeredgecolor=style.MARKER_EDGE,
                      markeredgewidth=0.4, markersize=4, label=SOURCE_LABELS[s])
               for s in SOURCE_ORDER]
    fig.legend(handles=handles, ncol=2, frameon=False, handletextpad=0.3,
               columnspacing=1.6, **kw)


def build_grid(df, name, plt, early="005000"):
    rows = [(early, f"{int(early)//1000}k steps"), ("030000", "30k steps")]
    fig, axes = plt.subplots(2, 3, figsize=style.figsize("text", ratio=0.70))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.925, bottom=0.10,
                        wspace=0.10, hspace=0.10)
    for j, (key, label) in enumerate(MODELS):
        sub_all = df[df["model"] == key]
        for i, (ckpt, _) in enumerate(rows):
            panel(axes[i][j], sub_all[sub_all["checkpoint"] == ckpt], 1.1)
        axes[0][j].set_title(label, fontsize=_fs(), pad=4)
    for i, (_, label) in enumerate(rows):
        axes[i][0].set_ylabel(label, fontsize=_fs())
    legend(fig, plt, loc="lower center", bbox_to_anchor=(0.52, -0.012))
    style.save(fig, name)


def build_single(df, key, name, plt, early="001000"):
    sub_all = df[df["model"] == key]
    cols = [(early, f"{int(early)//1000}k steps"), ("030000", "30k steps")]
    fig, axes = plt.subplots(1, 2, figsize=style.figsize("text", ratio=0.50))
    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.135, wspace=0.08)
    for ax, (ckpt, label) in zip(axes, cols):
        panel(ax, sub_all[sub_all["checkpoint"] == ckpt], 1.6)
        ax.set_title(label, fontsize=_fs(), pad=4)
    legend(fig, plt, loc="lower center", bbox_to_anchor=(0.52, -0.015))
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
    SOURCE_COLORS.update({"robot_3cam": style.PALETTE[0], "video_2cam": style.PALETTE[2]})
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv, dtype={"checkpoint": str})
    build_grid(df, f"umap_hardware{suffix}", plt)
    build_single(df, "sharedlam2", f"umap_hardware_sharedlam2{suffix}", plt)


if __name__ == "__main__":
    main()
