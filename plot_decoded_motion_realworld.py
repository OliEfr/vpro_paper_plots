r"""Human video -> robot action: decoded end-effector motion on real-world episodes.

Reads ``results/decoded_motion_realworld.csv`` and the key frames under
``results/frames_realworld/`` (both written by
``experiments/extract_decoded_motion_realworld.py``) and emits
``figures/decoded_motion_realworld[_science].pdf``.

What is shown. A linear (ridge) probe is fitted on the ROBOT episodes of the
real-world dataset, mapping the frozen 8-D latent to the 7-D end-effector target
(motion or absolute pose, see --target).
The same probe is then applied to the latents of HUMAN video, which has no action
labels. Each row is one episode: five key frames of the front camera on the left,
and on the right the decoded end-effector MOTION over the LAM horizon
(state[t+5]-state[t], i.e. x/y/z displacement over 5 frames = 0.17 s), one line per
LAM (multi-view front+side, single-view side-only). ``--target abs`` decodes the
stored DK1 action instead, which is an absolute end-effector pose target (it
tracks the measured pose about 5 frames ahead); that variant is saved with an
``_abs`` suffix.
The first row is a robot episode of the same task and carries the ground-truth
action as a black reference; on the human rows only the decoded curves exist, so
the figure argues by consistency with the visible motion and with the robot row,
not by a number.

The probe is fitted on robot latents only, so nothing about human video enters
the fit. Curves are smoothed with a centred 15-frame (0.5 s) moving average for
legibility; the raw per-frame values are in the CSV.

Usage:
    python plot_decoded_motion_realworld.py [--style paper|science]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style
import plot_probe_combo as combo

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "decoded_motion_realworld.csv"
FRAMES = HERE / "results" / "frames_realworld"

LAMS = [("ours_multi", "multi-view LAM (front + side)"), ("ours_single", "single-view LAM (side)")]
DIMS = {"abs": [("x", "$x$ [m]"), ("y", "$y$ [m]"), ("z", "$z$ [m]")],
        "delta": [("dx", r"$\Delta x$ [m / 5 frames]"), ("dy", r"$\Delta y$ [m / 5 frames]"),
                  ("dz", r"$\Delta z$ [m / 5 frames]")]}
GT_LABEL = {"abs": "ground-truth EE target (robot only)", "delta": "ground-truth EE motion (robot only)"}
N_FRAMES = 5
SMOOTH = 15


def smooth(v, k=SMOOTH):
    v = np.asarray(v, dtype=float)
    out = pd.Series(v).rolling(k, center=True, min_periods=1).mean().to_numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--probe", choices=["ridge", "mlp"], default="ridge")
    p.add_argument("--gallery", action="store_true",
                   help="read results/decoded_motion_realworld_gallery.csv (2 robot + 2 human episodes per task, "
                        "experiments/extract_decoded_motion_realworld.py --gallery) -> *_gallery figure")
    p.add_argument("--selected-train", action="store_true",
                   help="same as --selected for the best LAM-TRAINING tasks (*_selected_train.csv) -> *_selected_train figure")
    p.add_argument("--selected", action="store_true",
                   help="read results/decoded_motion_realworld_selected.csv (eval-task episodes ranked by the "
                        "ridge/MLP-agreement reliability proxy, select_decoded_motion_realworld.py) -> *_selected figure")
    p.add_argument("--target", choices=["delta", "abs"], default="delta",
                   help="delta: EE motion over the LAM horizon (state[t+5]-state[t]); abs: stored EE pose target")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.image import imread
    from matplotlib.ticker import MaxNLocator

    variant = "_gallery" if a.gallery else "_selected" if a.selected else "_selected_train" if a.selected_train else ""
    df = pd.read_csv(CSV.with_name(CSV.stem + variant + ".csv") if variant else CSV)
    episodes = df.drop_duplicates("episode_index")[["episode_index", "task", "source_kind", "row_label"]]
    episodes = episodes.sort_values("episode_index", key=lambda s: s.map(
        {e: i for i, e in enumerate(df.episode_index.unique())}))
    n_rows = len(episodes)

    w = style.TEXT_WIDTH
    h = w * 0.145 * n_rows + 0.35
    top = 1 - 0.7 / h  # keep the legend band a fixed height regardless of row count
    fig = plt.figure(figsize=(w, h))
    outer = GridSpec(n_rows, 2, figure=fig, left=0.06 if variant else 0.045, right=0.995, top=top, bottom=0.6 / h,
                     wspace=0.06, hspace=0.35, width_ratios=[N_FRAMES * 1.0, 3 * 1.55])
    gs_frames = [outer[r, 0].subgridspec(1, N_FRAMES, wspace=0.12) for r in range(n_rows)]
    gs_traces = [outer[r, 1].subgridspec(1, 3, wspace=0.42) for r in range(n_rows)]

    colors = {"ours_multi": combo.COLORS["ours_multi"], "ours_single": combo.COLORS["ours_single"]}
    fs = plt.rcParams["font.size"]
    row_axes = []
    for r, (_, ep) in enumerate(episodes.iterrows()):
        sub = df[df.episode_index == ep.episode_index].sort_values("frame_index")
        n = len(sub)
        idx = [int(round(k * (n - 1))) for k in (0, 0.25, 0.5, 0.75, 0.97)]
        for c, fi in enumerate(idx):
            ax = fig.add_subplot(gs_frames[r][0, c])
            if c == 0:
                row_axes.append(ax)
            img_path = FRAMES / f"ep{ep.episode_index}_front_f{fi}.jpg"
            ax.imshow(imread(img_path))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if c == 0:
                ax.set_ylabel(ep.row_label, fontsize=fs * (0.8 if n_rows > 6 else 1.0), labelpad=3)
            if r == 0:
                ax.set_title(["start", "25 %", "50 %", "75 %", "end"][c], fontsize=fs, pad=2)
        t = sub.frame_index.to_numpy() / 30.0
        for c, (dk, dlabel) in enumerate(DIMS[a.target]):
            ax = fig.add_subplot(gs_traces[r][0, c])
            if ep.source_kind == "robot_3cam":
                ax.plot(t, smooth(sub[f"gt_{dk}"]), color=style.INK, linewidth=0.9, label="ground truth")
            for lk, llabel in LAMS:
                ax.plot(t, smooth(sub[f"{lk}_{a.probe}_{dk}"]), color=colors[lk], linewidth=0.9, label=llabel)
            ax.axhline(0, color=style.INK_MUTED, linewidth=0.4, zorder=0)
            ax.grid(color=style.GRID, linewidth=0.4)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=fs * 0.85, pad=1.5)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if r == 0:
                ax.set_title(dlabel, fontsize=fs, pad=2)
            if r == n_rows - 1:
                ax.set_xlabel("time [s]", fontsize=fs, labelpad=1)
            else:
                ax.set_xticklabels([])
    from matplotlib.lines import Line2D
    if variant:  # thin rule between the task blocks (2*top_k rows each)
        blk = n_rows // 4
        for r in range(blk, n_rows, blk):
            y = (row_axes[r - 1].get_position().y0 + row_axes[r].get_position().y1) / 2
            fig.add_artist(Line2D([0.01, 0.99], [y, y], transform=fig.transFigure,
                                  color=style.INK_MUTED, linewidth=0.6))
    handles = [Line2D([0], [0], color=style.INK, linewidth=1.0, label=GT_LABEL[a.target])]
    handles += [Line2D([0], [0], color=colors[lk], linewidth=1.0, label=f"decoded, {llabel}") for lk, llabel in LAMS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995),
               columnspacing=1.6, handletextpad=0.5)
    suffix = "_science" if a.style == "science" else ""
    name = f"decoded_motion_realworld{variant}{'' if a.probe == 'ridge' else '_mlp'}{'' if a.target == 'delta' else '_abs'}{suffix}"
    style.save(fig, name)


if __name__ == "__main__":
    main()
