r"""Human -> robot transfer, before / after (real-world DK1, training tasks and eval tasks).

Reads ``results/xemb_human2robot.csv`` and the frames in ``results/frames_xemb_human2robot/`` (written by
``experiments/extract_xemb_human2robot.py dump``) and emits ``figures/xemb_human2robot[_science].{pdf,png}``.

What is shown. Each row is one example. Left pair: a real HUMAN demo, before and 0.6 s later; the arrow is
the hand's image motion (optical flow, drawn at 2x). Right pair: a ROBOT start frame of the same task, and
the LAM decoder's prediction after applying the human's own latents (two motion latents, one per 0.3 s);
the arrow is the predicted motion. The robot "after" frame is model output (hence blurry); nothing is
post-processed. Left block: tasks seen in training (not eval tasks); right block: real-hardware eval tasks.
Cherry-picked from ``figures/xemb_human2robot_gallery_{train,eval}.jpg``; the direction agreement of every
candidate tried is in ``results/xemb_human2robot_candidates.csv``.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_human2robot.pdf}
      \caption{Human $\rightarrow$ robot transfer through the latent action (real world, multi-view LAM).
        Each row: a human demonstration before and 0.6\,s later (left), and a robot start frame of the same
        task with the human's latents applied by the LAM decoder (right). Arrows: image motion ($2\times$).
        Left: training tasks; right: evaluation tasks.}
      \label{fig:xemb_human2robot}
    \end{figure*}

Usage:
    python plot_xemb_human2robot.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "xemb_human2robot.csv"
FRAMES = HERE / "results" / "frames_xemb_human2robot"
ARROW_GAIN = 2.0
SPLITS = [("train", "training tasks"), ("eval", "evaluation tasks")]


def arrow(ax, cx, cy, u, v, color):
    import matplotlib.patheffects as pe
    u, v = ARROW_GAIN * u, ARROW_GAIN * v
    a = ax.annotate("", xy=(cx + u / 2, cy + v / 2), xytext=(cx - u / 2, cy - v / 2), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color=color, linewidth=1.3,
                                    shrinkA=0, shrinkB=0))
    a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
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
    nrow = max((d.split == s).sum() for s, _ in SPLITS)
    w = style.TEXT_WIDTH
    gap, mid, block_gap, left = 0.02, 0.10, 0.26, 0.04
    tile = (w - 2 * left - block_gap - 2 * (2 * gap + mid)) / 8
    top, row_gap, bottom = 0.52, 0.17, 0.06
    h = top + nrow * tile + nrow * row_gap + bottom
    fig = plt.figure(figsize=(w, h))
    cols = [("human_before", "human"), ("human_after", "+0.6 s"), ("robot_before", "robot"), ("robot_after_model", "+ human latents")]
    for bi, (split, title) in enumerate(SPLITS):
        x0 = left + bi * (4 * tile + 2 * gap + mid + block_gap)
        xs = [x0, x0 + tile + gap, x0 + 2 * tile + gap + mid, x0 + 3 * tile + 2 * gap + mid]
        fig.text((x0 + 2 * tile + gap + mid / 2) / w, 1 - 0.10 / h, title, ha="center", va="top", fontsize=fs)
        for k, (_, lab) in enumerate(cols):
            fig.text((xs[k] + tile / 2) / w, 1 - (top - 0.05) / h, lab, ha="center", va="bottom", fontsize=fs - 1,
                     color=style.INK if k % 2 == 0 else style.INK_MUTED)
        g = d[d.split == split].reset_index(drop=True)
        for ri, r in g.iterrows():
            y = top + ri * (tile + row_gap)
            for k, (tag, _) in enumerate(cols):
                ax = fig.add_axes([xs[k] / w, 1 - (y + tile) / h, tile / w, tile / h])
                ax.imshow(imread(FRAMES / f"ex{int(r.example)}_{tag}.png"))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(tag == "robot_after_model")
                    s.set_color(style.INK); s.set_linewidth(0.8)
                if tag == "human_after":
                    arrow(ax, r.human_cx, 1 - r.human_cy, r.human_u, -r.human_v, style.INK)
                if tag == "robot_after_model":
                    arrow(ax, r.robot_cx, 1 - r.robot_cy, r.robot_u, -r.robot_v, style.INK)
            ax_mid = (xs[1] + tile + mid / 2) / w
            fig.text(ax_mid, 1 - (y + tile / 2) / h, r"$\rightarrow$", ha="center", va="center", fontsize=fs + 2)
            task = r.task_human if r.task_human == r.task_robot else f"{r.task_human} / {r.task_robot}"
            fig.text((x0 + 2 * tile + gap + mid / 2) / w, 1 - (y + tile + 0.03) / h, task, ha="center", va="top",
                     fontsize=fs - 1.5, color=style.INK_MUTED)
            print(f"{split:5s} ex{int(r.example)}  {task:40s} cos {r.cos_direction:.2f}")
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_human2robot{suffix}")


if __name__ == "__main__":
    main()
