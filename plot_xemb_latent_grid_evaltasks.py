r"""One latent, one movement -- on the robot and on the human (LAPA-style latent-action grid).

Reads ``results/xemb_latent_grid.csv`` and the cells in ``results/frames_xemb_latent_grid/``
(written by ``experiments/extract_xemb_evaltasks.py grid``; EVAL TASKS ONLY) and emits
``figures/xemb_latent_grid_evaltasks[_science].{pdf,png}``.

What is shown. Real-world DK1, the paper's multi-view LAM. Each COLUMN is one latent action:
the latent of a single real moment of a demo (0.3 s; top label says whether that demo was a
robot or a human), reduced to its motion part (minus the latent of the same frame held still).
Each ROW is a real start frame -- two robot, two human -- and every cell is what the LAM's
decoder predicts when that one latent is applied to that frame (two steps, ~0.6 s). The same
latent moves the gripper and the hand the same way; the bottom label names the direction in
the image. Cherry-picked by design; ``results/xemb_latent_grid_stats.csv`` carries how
consistent each latent is over 30 start frames of both embodiments.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_latent_grid_evaltasks.pdf}
      \caption{One latent action, one movement, across embodiments (real world, multi-view
        LAM). Each column applies a single latent, taken from one moment of a robot or a
        human demonstration, to four real start frames (left); the LAM decoder's prediction
        moves the robot gripper and the human hand in the same direction.}
      \label{fig:xemb_latent_grid_evaltasks}
    \end{figure*}

Usage:
    python plot_xemb_latent_grid_evaltasks.py [--style paper|science]
"""
import argparse
from pathlib import Path

import pandas as pd

import style

HERE = Path(__file__).resolve().parent
META = HERE / "results" / "xemb_latent_grid_evaltasks.csv"
CELLS = HERE / "results" / "frames_xemb_latent_grid_evaltasks"
DIRECTION = {"right": "right", "up-right": "up, right", "away": "away", "up-left": "up, left",
             "left": "left", "down-left": "down, left", "down-right": "down, right"}


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

    meta = pd.read_csv(META)
    rows = meta[meta.kind == "row"].sort_values("index")
    cols = meta[meta.kind == "col"].sort_values("index")
    ncol = len(cols) + 1

    w = style.TEXT_WIDTH
    left, right = 0.34, 0.04            # inch; left margin holds the row labels
    gap_c, gap_r, gap_emb = 0.03, 0.03, 0.10
    first_gap = 0.10                    # extra space after the start-frame column
    tile = (w - left - right - first_gap - gap_c * (ncol - 1)) / ncol
    top, bottom = 0.30, 0.20
    nrow = len(rows)
    emb_break = int((rows.role == "robot").sum())
    h = top + nrow * tile + (nrow - 1) * gap_r + gap_emb + bottom
    fig = plt.figure(figsize=(w, h))
    fs = plt.rcParams["font.size"]

    def x_of(ci):
        return left + ci * (tile + gap_c) + (first_gap if ci > 0 else 0)

    def y_of(ri):
        return top + ri * (tile + gap_r) + (gap_emb if ri >= emb_break else 0)

    print(f"{'cell':8s} source")
    for ri, r in enumerate(rows.itertuples()):
        for ci in range(ncol):
            ax = fig.add_axes([x_of(ci) / w, 1 - (y_of(ri) + tile) / h, tile / w, tile / h])
            ax.imshow(imread(CELLS / f"r{ri}_c{ci}.png"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(ci == 0)
                s.set_color(style.INK); s.set_linewidth(0.8)
            if ci == 0:
                ax.set_ylabel("Robot" if r.role == "robot" else "Human", labelpad=2)
        print(f"row {ri}   {r.role} ep {r.episode} frame {r.frame} ({r.task})")

    # column labels: top = where the latent comes from, bottom = what it does
    fig.text((x_of(0) + tile / 2) / w, 1 - (top - 0.05) / h, "start frame", ha="center", va="bottom", fontsize=fs)
    for ci, c in enumerate(cols.itertuples(), start=1):
        xc = (x_of(ci) + tile / 2) / w
        fig.text(xc, 1 - (top - 0.05) / h, f"latent from {c.role}", ha="center", va="bottom", fontsize=fs - 1,
                 color=style.INK_MUTED)
        fig.text(xc, 1 - (y_of(nrow - 1) + tile + 0.05) / h, DIRECTION.get(c.label, c.label), ha="center", va="top",
                 fontsize=fs)
        print(f"col {ci}   {c.label:10s} from {c.role} ep {c.episode} frame {c.frame} ({c.task})")
    fig.text((x_of(0) + tile / 2) / w, 1 - (y_of(nrow - 1) + tile + 0.05) / h, "movement:", ha="center", va="top",
             fontsize=fs)

    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_latent_grid_evaltasks{suffix}")


if __name__ == "__main__":
    main()
