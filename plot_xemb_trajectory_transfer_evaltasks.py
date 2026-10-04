r"""Trajectory transfer: a demo's latent SEQUENCE drives the other embodiment (real-world DK1).

Reads ``results/xemb_trajectory_transfer.csv`` and the frames in ``results/frames_xemb_trajectory/``
(written by ``experiments/extract_xemb_evaltasks.py traj``; EVAL TASKS ONLY) and emits
``figures/xemb_trajectory_transfer_evaltasks[_science].{pdf,png}``.

What is shown. Two runs side by side; columns are time (0 to 1.2 s, one motion latent every 0.3 s).
Top row: the real demo the latents come from. Middle: a start frame of the OTHER embodiment driven
only by that latent sequence: the LAM decoder's own output (raw reconstruction, 256 px, hence
blurry; the 0 s frame is the real start frame). Bottom: what that embodiment really did in its own demo of the same movement
(reference, never seen by the model). Scores per step (flow direction of the generated motion vs
the source's and the reference's real motion) are in the csv and printed.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_trajectory_transfer_evaltasks.pdf}
      \caption{A demonstration's latent sequence drives the other embodiment (real world,
        multi-view LAM). Top: real demo providing one latent every 0.3\,s. Middle: a start
        frame of the other embodiment driven only by these latents (LAM decoder output).
        Bottom: that embodiment's own real demo of the same
        movement, for reference. Left: human $\rightarrow$ robot (milk on plate); right:
        robot (salt on plate) $\rightarrow$ human (banana in bowl).}
      \label{fig:xemb_trajectory_transfer_evaltasks}
    \end{figure*}

Usage:
    python plot_xemb_trajectory_transfer_evaltasks.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "xemb_trajectory_transfer_evaltasks.csv"
FRAMES = HERE / "results" / "frames_xemb_trajectory_evaltasks"
RUNS = [("milk_on_plate", "human_to_robot", "human $\\rightarrow$ robot: milk on plate"),
        ("salt_robot_banana_human", "robot_to_human", "robot (salt) $\\rightarrow$ human (banana)")]
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

    sc = pd.read_csv(CSV)
    fs = plt.rcParams["font.size"]
    w = style.TEXT_WIDTH
    left, gap, block_gap, right = 0.55, 0.02, 0.62, 0.04
    tile = (w - left - right - block_gap - 2 * K * gap) / (2 * (K + 1))
    th = tile * 3 / 4
    top, bottom, rgap = 0.36, 0.20, 0.03
    h = top + 3 * th + 2 * rgap + bottom
    fig = plt.figure(figsize=(w, h))
    for bi, (name, direction, title) in enumerate(RUNS):
        src, tgt = direction.split("_to_")
        x0 = left + bi * ((K + 1) * tile + K * gap + block_gap)
        fig.text((x0 + ((K + 1) * tile + K * gap) / 2) / w, 1 - 0.12 / h, title, ha="center", va="top", fontsize=fs)
        rows = [("source", f"real\n{src}"), ("decoded", f"{tgt} +\nlatents"), ("target_real", f"real\n{tgt}")]
        for ri, (tag, lab) in enumerate(rows):
            y = top + ri * (th + rgap)
            for k in range(K + 1):
                ax = fig.add_axes([(x0 + k * (tile + gap)) / w, 1 - (y + th) / h, tile / w, th / h])
                ax.imshow(imread(FRAMES / f"{name}_{direction}_{tag}_k{k}.png"), aspect="auto")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(tag == "decoded")
                    s.set_color(style.INK); s.set_linewidth(0.8)
                if k == 0:
                    ax.set_ylabel(lab, labelpad=2, fontsize=fs - 0.5)
                if ri == 2:
                    ax.set_xlabel(f"{k * 0.3:.1f} s", labelpad=1.5, fontsize=fs - 1)
        g = sc[(sc.pair == name) & (sc.direction == direction)]
        print(f"{name} {direction}: mean cos vs source {g.cos_generated_vs_source_real.mean():.3f}, "
              f"vs target real future {g.cos_generated_vs_target_real_future.mean():.3f}")
    print("\nall runs:")
    print(sc.groupby(["pair", "direction"])[["cos_generated_vs_source_real", "cos_generated_vs_target_real_future"]]
          .mean().round(3).to_string())
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_trajectory_transfer_evaltasks{suffix}")


if __name__ == "__main__":
    main()
