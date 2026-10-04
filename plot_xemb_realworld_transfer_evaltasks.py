r"""Same movement, same latent -- different movement, different latent: human <-> robot on DK1.

Reads the dumps written by ``experiments/extract_xemb_evaltasks.py pairs_dump`` (EVAL TASKS ONLY):
``results/xemb_realworld_transfer.csv`` (latent sequences), ``..._arrows.csv`` (motion
arrows), ``..._similarity.csv`` (clip-by-clip latent similarity) and the key frames in
``results/frames_xemb_realworld/``. Emits ``figures/xemb_realworld_transfer_evaltasks[_science].{pdf,png}``.

What is shown. Real-world DK1, the paper's multi-view LAM. Four different movements (rows).
For each, a ROBOT clip and the HUMAN clip whose latent sequence is its nearest neighbour,
1 s each: first and last front-camera frame, the image motion of gripper / hand as an arrow
(optical flow, drawn at 2x length), and the two latent sequences as heatmaps (8 latent dims
x 30 frames). The heatmaps show each latent minus its embodiment's average latent: robot
and human latents differ by a constant offset, which says "robot" or "human" and nothing
about the movement; removing it leaves the movement. Right: correlation of these latent
sequences between all eight clips -- high within a movement (robot and human agree),
low across movements (the latents are different for different movements).

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_realworld_transfer_evaltasks.pdf}
      \caption{Same movement, same latent across embodiments (real world, multi-view LAM).
        Left: four movements, each a robot clip and the human clip with the nearest latent
        sequence (1\,s, first and last frame; arrows: image motion, $2\times$); heatmaps:
        their latent sequences, each minus its embodiment's average latent. Right:
        correlation of the latent sequences of all eight clips -- high within a movement,
        low across movements.}
      \label{fig:xemb_realworld_transfer_evaltasks}
    \end{figure*}

Usage:
    python plot_xemb_realworld_transfer_evaltasks.py [--style paper|science]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
LAT = HERE / "results" / "xemb_realworld_transfer_evaltasks.csv"
ARR = HERE / "results" / "xemb_realworld_transfer_evaltasks_arrows.csv"
SIM = HERE / "results" / "xemb_realworld_transfer_evaltasks_similarity.csv"
FRAMES = HERE / "results" / "frames_xemb_realworld_evaltasks"
MOVES = [("right", "right"), ("up_right", "up-right"), ("left", "left"), ("down", "down")]
ROLES = [("robot", "Robot"), ("human", "Human")]
CROP_FRAC = 0.58          # crop side, fraction of frame width / height, centred on the moving gripper / hand
ARROW_GAIN = 2.0          # arrows drawn at 2x their measured length, for legibility at print size


def crop_box(cx, cy):
    h = CROP_FRAC / 2
    return float(np.clip(cx - h, 0, 1 - CROP_FRAC)), float(np.clip(cy - h, 0, 1 - CROP_FRAC))


def show(ax, img, box):
    H, Wd = img.shape[:2]
    x0, y0 = box
    ax.imshow(img[int(y0 * H):int((y0 + CROP_FRAC) * H), int(x0 * Wd):int((x0 + CROP_FRAC) * Wd)],
              aspect="auto", extent=(x0, x0 + CROP_FRAC, y0 + CROP_FRAC, y0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def arrow(ax, r, color):
    import matplotlib.patheffects as pe
    u, v = ARROW_GAIN * r.u, ARROW_GAIN * r.v
    x, y = r.cx - u / 2, r.cy - v / 2
    a = ax.annotate("", xy=(x + u, y + v), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color=color,
                                    linewidth=1.4, shrinkA=0, shrinkB=0))
    a.arrow_patch.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
    a.arrow_patch.set_clip_path(ax.patch)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.patches import Rectangle

    lat, arr = pd.read_csv(LAT), pd.read_csv(ARR)
    sim = pd.read_csv(SIM, index_col=0)
    zc = [f"zc{k}" for k in range(8)]
    vmax = np.abs(lat[zc].values).max()
    fs = plt.rcParams["font.size"]

    w = style.TEXT_WIDTH
    tile_w = 0.66
    tile_h = tile_w * 3 / 4
    label_w, gap_t, gap_emb, heat_gap, heat_w = 0.66, 0.02, 0.10, 0.10, 0.78
    top, bottom, row_gap = 0.32, 0.20, 0.09
    h = top + len(MOVES) * tile_h + (len(MOVES) - 1) * row_gap + bottom
    fig = plt.figure(figsize=(w, h))
    X = lambda inch: inch / w
    Yb = lambda inch_from_top, height: 1 - (inch_from_top + height) / h   # bottom of an axes, from top offset

    def x_tile(role_i, k):
        return label_w + role_i * (2 * tile_w + gap_t + gap_emb) + k * (tile_w + gap_t)

    x_heat = x_tile(1, 1) + tile_w + heat_gap
    for ri, (role, rlabel) in enumerate(ROLES):
        fig.text(X(x_tile(ri, 0) + tile_w + gap_t / 2), 1 - (top - 0.06) / h, f"{rlabel}: start  /  after 1 s",
                 ha="center", va="bottom", fontsize=fs)
    fig.text(X(x_heat + heat_w / 2), 1 - (top - 0.06) / h, "latent over 1 s", ha="center", va="bottom", fontsize=fs)

    print(f"{'movement':11s} {'role':6s} episode frame  task")
    for mi, (mv, mlabel) in enumerate(MOVES):
        y_top = top + mi * (tile_h + row_gap)
        fig.text(X(label_w - 0.06), Yb(y_top, tile_h / 2), mlabel, ha="right", va="center", fontsize=fs)
        for ri, (role, rlabel) in enumerate(ROLES):
            r_arr = arr[(arr.panel == mv) & (arr.role == role)].iloc[0]
            box = crop_box(r_arr.cx, r_arr.cy)
            for k, kf in enumerate((0, 2)):
                ax = fig.add_axes([X(x_tile(ri, k)), Yb(y_top, tile_h), X(tile_w), tile_h / h])
                show(ax, imread(FRAMES / f"{mv}_{role}_k{kf}.jpg"), box)
                if k == 1:
                    arrow(ax, r_arr, style.INK)
            zz = lat[(lat.pair == mv) & (lat.role == role)].sort_values("step")
            hh = (tile_h - 0.03) / 2
            axh = fig.add_axes([X(x_heat), Yb(y_top + ri * (hh + 0.03), hh), X(heat_w), hh / h])
            axh.imshow(zz[zc].values.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
            axh.set_xticks([]); axh.set_yticks([])
            axh.minorticks_off()
            for s in axh.spines.values():
                s.set_visible(False)
            axh.text(-0.04, 0.5, rlabel[0], transform=axh.transAxes, ha="right", va="center", fontsize=fs - 1)
            e0 = zz.iloc[0]
            print(f"{mv:11s} {role:6s} {int(e0.episode):7d} {int(e0.frame):5d}  {e0.task}")

    # similarity matrix
    names = list(sim.index)
    m_left = x_heat + heat_w + 0.80
    m_size = min(w - m_left - 0.08, len(MOVES) * (tile_h + row_gap) - row_gap)
    axm = fig.add_axes([X(m_left), Yb(top, m_size), X(m_size), m_size / h])
    axm.imshow(sim.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    short = {mv: lab for mv, lab in MOVES}
    labels = [f"{short[n.split(':')[0]]} ({n.split(':')[1][0].upper()})" for n in names]
    axm.set_xticks(range(len(names))); axm.set_yticks(range(len(names)))
    axm.set_xticklabels([n.split(':')[1][0].upper() for n in names], fontsize=fs - 1.5)   # R / H; rows carry the names
    axm.set_yticklabels(labels, fontsize=fs - 1.5)
    axm.minorticks_off()
    axm.tick_params(which="both", length=0, pad=1.5, top=False, right=False)
    for s in axm.spines.values():
        s.set_visible(False)
    for i in range(0, len(names), 2):
        axm.add_patch(Rectangle((i - 0.5, i - 0.5), 2, 2, fill=False, edgecolor=style.INK, linewidth=0.9))
    axm.set_title("latent similarity between clips", pad=3, fontsize=fs)

    print("\nlatent similarity (correlation of the centred sequences):")
    print(sim.round(2).to_string())
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_realworld_transfer_evaltasks{suffix}")


if __name__ == "__main__":
    main()
