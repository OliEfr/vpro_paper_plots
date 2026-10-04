r"""Human -> robot trajectory transfer, several examples per split (real-world DK1, training and eval tasks).

Reads ``results/xemb_traj_h2r.csv`` and ``results/frames_xemb_traj_h2r/`` (written by
``experiments/extract_xemb_traj_h2r.py dump``) and emits ``figures/xemb_traj_h2r[_science].{pdf,png}``.

What is shown. Each example is a three-row strip over 1.2 s (columns: 0, 0.3, 0.6, 0.9, 1.2 s).
Top: a real HUMAN demo, the source of one motion latent per 0.3 s. Middle (framed): a ROBOT start frame driven
only by those latents through the LAM decoder (raw model output). Bottom: that robot's own real demo of the same
movement (reference; the model never sees it). Left: training tasks; right: evaluation tasks. Cherry-picked
from ``figures/xemb_traj_h2r_gallery_{train,eval}.jpg``; scores of every run tried are in
``results/xemb_traj_h2r_candidates.csv``.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_traj_h2r.pdf}
      \caption{Human $\rightarrow$ robot trajectory transfer (real world, multi-view LAM). Each strip: a human
        demonstration providing one latent every 0.3\,s (top), a robot start frame driven only by these latents
        through the LAM decoder (middle), and the robot's own demonstration of the same movement (bottom,
        reference). Left: training tasks; right: evaluation tasks.}
      \label{fig:xemb_traj_h2r}
    \end{figure*}

Usage:
    python plot_xemb_traj_h2r.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "xemb_traj_h2r.csv"
FRAMES = HERE / "results" / "frames_xemb_traj_h2r"
SPLITS = [("train", "training tasks"), ("eval", "evaluation tasks")]
ROWS = [("human", "human"), ("robot_model", "robot +\nlatents"), ("robot_real", "robot\n(real)")]
K = 4


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
    nex = max((d.split == s).sum() for s, _ in SPLITS)
    w = style.TEXT_WIDTH
    label_w, gap, block_gap, right = 0.42, 0.015, 0.50, 0.03
    tile = (w - 2 * label_w - block_gap - right - 2 * K * gap) / (2 * (K + 1))
    rgap, ex_gap, top, bottom = 0.012, 0.20, 0.36, 0.20
    ex_h = 3 * tile + 2 * rgap
    h = top + nex * ex_h + (nex - 1) * ex_gap + bottom
    fig = plt.figure(figsize=(w, h))
    for bi, (split, title) in enumerate(SPLITS):
        x0 = label_w + bi * (label_w + (K + 1) * tile + K * gap + block_gap)
        fig.text((x0 + ((K + 1) * tile + K * gap) / 2) / w, 1 - 0.10 / h, title, ha="center", va="top", fontsize=fs)
        g = d[d.split == split].reset_index(drop=True)
        for ei, r in g.iterrows():
            y0 = top + ei * (ex_h + ex_gap)
            task = r.task_human if r.task_human == r.task_robot else f"{r.task_human} / {r.task_robot}"
            fig.text((x0 + ((K + 1) * tile + K * gap) / 2) / w, 1 - (y0 - 0.025) / h, task, ha="center", va="bottom",
                     fontsize=fs - 1.5, color=style.INK_MUTED)
            for ri, (tag, lab) in enumerate(ROWS):
                y = y0 + ri * (tile + rgap)
                for k in range(K + 1):
                    ax = fig.add_axes([(x0 + k * (tile + gap)) / w, 1 - (y + tile) / h, tile / w, tile / h])
                    ax.imshow(imread(FRAMES / f"ex{int(r.example)}_{tag}_k{k}.png"))
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(tag == "robot_model")
                        s.set_color(style.INK); s.set_linewidth(0.8)
                    if k == 0:
                        ax.set_ylabel(lab, labelpad=2, fontsize=fs - 1.5)
                    if ri == 2 and ei == len(g) - 1:
                        ax.set_xlabel(f"{k * 0.3:.1f} s", labelpad=1.5, fontsize=fs - 1.5)
            print(f"{split:5s} ex{int(r.example)} {task:45s} cos vs human {r.cos_vs_human:.2f}, vs robot real {r.cos_vs_robot_real:.2f}")
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_traj_h2r{suffix}")


if __name__ == "__main__":
    main()
