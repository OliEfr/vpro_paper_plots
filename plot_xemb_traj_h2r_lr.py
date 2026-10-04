r"""Human -> robot trajectory transfer, left/right movements, original pick-and-place eval tasks (real-world DK1).

Reads ``results/xemb_traj_h2r_lr.csv`` and ``results/frames_xemb_traj_h2r_lr/`` (written by
``experiments/extract_xemb_traj_h2r_lr.py dump``) and emits ``figures/xemb_traj_h2r_lr[_science].{pdf,png}``.

What is shown. Each example is a three-row strip over 1.2 s (columns: 0, 0.3, 0.6, 0.9, 1.2 s). Top: a real
HUMAN demo, source of one motion latent per 0.3 s. Middle (framed): a ROBOT start frame driven only by those
latents, the LAM decoder's raw output (blurry; 0 s = real start frame). Bottom: that robot's own real demo,
which moves the same way (reference, never seen by the model). Only horizontal (left/right) movements, only
the four original pick-and-place eval tasks. Arrows on the 1.2 s frames: image motion over the strip (optical
flow; human / real robot: first -> last frame; model row: vs a zero-motion rollout); the arrow shows the
DIRECTION only (fixed length), placed at the moving gripper / hand.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_traj_h2r_lr.pdf}
      \caption{Human $\rightarrow$ robot transfer of left/right movements on the evaluation tasks (real world,
        multi-view LAM). Each strip: a human demonstration providing one latent every 0.3\,s (top), a robot
        start frame driven only by these latents through the LAM decoder (middle), and the robot's own
        demonstration of the same movement (bottom). Arrows: image motion.}
      \label{fig:xemb_traj_h2r_lr}
    \end{figure*}

Usage:
    python plot_xemb_traj_h2r_lr.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "xemb_traj_h2r_lr.csv"
FRAMES = HERE / "results" / "frames_xemb_traj_h2r_lr"
ROWS = [("human", "human"), ("robot_model", "robot +\nlatents"), ("robot_real", "robot\n(real)")]
K = 4
ARROW_LEN = 0.34   # fixed arrow length, fraction of the tile width (direction is what matters)


def arrow(ax, cx, cy, u, v, color):
    import matplotlib.patheffects as pe
    import numpy as np
    n = float(np.hypot(u, v)) or 1.0
    u, v = ARROW_LEN * u / n, ARROW_LEN * v / n
    cx = min(max(cx, abs(u) / 2 + 0.03), 1 - abs(u) / 2 - 0.03)
    cy = min(max(cy, abs(v) / 2 + 0.04), 1 - abs(v) / 2 - 0.04)
    a = ax.annotate("", xy=(cx + u / 2, 1 - (cy + v / 2)), xytext=(cx - u / 2, 1 - (cy - v / 2)), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.22", color=color, linewidth=1.5,
                                    shrinkA=0, shrinkB=0))
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

    d = pd.read_csv(CSV)
    fs = plt.rcParams["font.size"]
    ncol = 2
    nper = (len(d) + ncol - 1) // ncol
    w = style.TEXT_WIDTH
    label_w, gap, block_gap, right = 0.42, 0.015, 0.40, 0.03
    tile = (w - ncol * label_w - (ncol - 1) * block_gap - right - ncol * K * gap) / (ncol * (K + 1))
    th = tile * 3 / 4
    rgap, ex_gap, top, bottom = 0.012, 0.22, 0.14, 0.20
    ex_h = 3 * th + 2 * rgap
    h = top + nper * ex_h + (nper - 1) * ex_gap + bottom
    fig = plt.figure(figsize=(w, h))
    arrow_col = {"human": style.INK, "robot_model": style.INK, "robot_real": style.INK}
    for i, r in d.iterrows():
        col, ei = i % ncol, i // ncol
        x0 = label_w + col * (label_w + (K + 1) * tile + K * gap + block_gap)
        y0 = top + ei * (ex_h + ex_gap)
        task = r.task_human if r.task_human == r.task_robot else f"{r.task_human} / {r.task_robot}"
        dirtxt = r"$\rightarrow$ right" if r.direction == "right" else r"$\leftarrow$ left"
        fig.text((x0 + ((K + 1) * tile + K * gap) / 2) / w, 1 - (y0 - 0.025) / h, f"{task}  ({dirtxt})",
                 ha="center", va="bottom", fontsize=fs - 1.5, color=style.INK_MUTED)
        for ri, (tag, lab) in enumerate(ROWS):
            y = y0 + ri * (th + rgap)
            for k in range(K + 1):
                ax = fig.add_axes([(x0 + k * (tile + gap)) / w, 1 - (y + th) / h, tile / w, th / h])
                ax.imshow(imread(FRAMES / f"ex{int(r.example)}_{tag}_k{k}.png"), aspect="auto", extent=(0, 1, 0, 1))
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(tag == "robot_model")
                    s.set_color(style.INK); s.set_linewidth(0.8)
                if k == 0:
                    ax.set_ylabel(lab, labelpad=2, fontsize=fs - 1.5)
                if k == K:
                    arrow(ax, r[f"{tag}_cx"], r[f"{tag}_cy"], r[f"{tag}_u"], r[f"{tag}_v"], arrow_col[tag])
                if ri == 2 and ei == nper - 1:
                    ax.set_xlabel(f"{k * 0.3:.1f} s", labelpad=1.5, fontsize=fs - 1.5)
        print(f"ex{int(r.example)} {task:42s} {r.direction:5s} cos vs human {r.cos_vs_human:.2f}, vs robot real {r.cos_vs_robot_real:.2f}")
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_traj_h2r_lr{suffix}")


if __name__ == "__main__":
    main()
