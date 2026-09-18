r"""Real-world probing + decoded human-video motion in ONE double-column figure.

Combines the two real-world figures into a single ``figure*`` (IEEE double column,
height:width about 1:3):

  left   the MLP block of plot_probe_perdim_realworld.py: R^2 of the MLP(512,256) probe
         that decodes the 5-frame EE motion state[t+5]-state[t] from the frozen 8-D
         latent, single-view (side) vs multi-view (front + side) LAM, held-out robot
         episodes (results/probe_perdim_realworld.csv, target state_delta_h5)
  right  two rows of plot_decoded_motion_realworld.py --selected: one ROBOT and one
         HUMAN episode of the eval task "milk on pink plate" (the top-2-by-proxy
         episodes 596 / 1638 of results/decoded_motion_realworld_selection.csv), each
         with three front-camera key frames (start, 50 %, end) and the decoded
         Delta-x / Delta-y / Delta-z traces (cm over 5 frames) of both LAMs; the robot row carries the
         ground-truth motion in black. The ridge decoding is shown (as in the source
         figure); --probe mlp switches to the MLP decoding.

Data and probes are exactly those of the two source figures; nothing is refitted here.
Frames are cut by the same ffmpeg command as the other key frames
(experiments/extract_decoded_motion_realworld.py) from the staged front-camera videos.

Usage:
    python plot_realworld_probe_decoded.py [--style paper|science] [--probe ridge|mlp]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style
import plot_probe_combo as combo
from plot_probe_perdim_realworld import GROUPS, METHODS, load
from plot_decoded_motion_realworld import DIMS, FRAMES, LAMS, smooth

HERE = Path(__file__).resolve().parent
TRACES = HERE / "results" / "decoded_motion_realworld_selected.csv"
EPISODES = [(596, "Robot"), (1638, "Human")]  # milk on pink plate, rank 2 by proxy
KEYS = (0.0, 0.5, 0.97)
KEY_TITLES = ["start", "50 %", "end"]
BAR_W, GAP_MEAN = 0.36, 0.35
LEGEND = [("ours_single", "Ours (single-view)"), ("ours_multi", "Ours (multi-view)")]  # short, as in plot_probe_tsne_combo.py


def probe_panel(ax, r2):
    """MLP block of plot_probe_perdim_realworld.py, at body font size and without value labels
    (same look as the probe panel of plot_probe_tsne_combo.py)."""
    x0 = 0.0
    ticks, labels = [], []
    for gi, (gkey, glabel) in enumerate(GROUPS["state_delta_h5"]):
        for mi, (mkey, _) in enumerate(METHODS):
            xb = x0 + (mi - (len(METHODS) - 1) / 2) * BAR_W
            ax.bar(xb, r2[("mlp", mkey, gkey)], BAR_W, facecolor=combo.COLORS[mkey],
                   edgecolor=style.MARKER_EDGE, linewidth=0.5, zorder=3)
        ticks.append(x0)
        labels.append("grip" if gkey == "gripper" else glabel)  # short label, as in plot_probe_tsne_combo.py
        if gi == 0:
            ax.axvline(x0 + 0.5 + GAP_MEAN / 2, color=style.INK_MUTED, linewidth=0.6, linestyle="--", zorder=2)
            x0 += 1 + GAP_MEAN
        else:
            x0 += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x0 - 1 + 0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MLP $R^2$")
    ax.set_xlabel("Action Dimension", labelpad=1)
    ax.grid(axis="y", color=style.GRID, linewidth=0.5, zorder=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", choices=["paper", "science"], default="paper")
    p.add_argument("--probe", choices=["ridge", "mlp"], default="ridge")
    a = p.parse_args()
    global style
    if a.style == "science":
        import style_science
        style = style_science
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    r2 = load("state_delta_h5")
    df = pd.read_csv(TRACES)
    colors = {k: combo.COLORS[k] for k, _ in LAMS}

    w = style.TEXT_WIDTH
    h = w * 0.26
    fig = plt.figure(figsize=(w, h))
    # Explicit geometry (figure fractions) so that (i) every image box has the video's 4:3
    # aspect, (ii) the trace axes of a row have exactly the image height, and (iii) the two
    # rows together span the probe axes' vertical extent (top of row 0 = top of the bars,
    # bottom of row 1 = bottom of the bars).
    L, R, B, T = 0.06, 0.995, 0.17, 0.80
    PROBE_W, SEP_GAP = 0.265, 0.025         # probe axes width; gap either side of the vertical rule
    ROW_GAP, FRAME_GAP, TRACE_GAP, BLOCK_GAP = 0.10, 0.008, 0.03, 0.04
    ax_probe = fig.add_axes([L, B, PROBE_W, T - B])
    probe_panel(ax_probe, r2)
    x_sep = L + PROBE_W + SEP_GAP
    # separator over the plotting region plus the tick-label band, not the full figure height
    fig.add_artist(Line2D([x_sep, x_sep], [B - 0.07, T], transform=fig.transFigure,
                          color=style.INK_MUTED, linewidth=0.6))
    x_right = x_sep + SEP_GAP
    row_h = (T - B - ROW_GAP) / len(EPISODES)
    img_w = row_h * h * (4 / 3) / w            # 4:3 video frame, in figure-width fractions
    frames_w = len(KEYS) * img_w + (len(KEYS) - 1) * FRAME_GAP
    x_traces = x_right + frames_w + BLOCK_GAP
    trace_w = (R - x_traces - 2 * TRACE_GAP) / 3

    for r, (ep, klabel) in enumerate(EPISODES):
        sub = df[df.episode_index == ep].sort_values("frame_index")
        assert len(sub), f"episode {ep} not in {TRACES.name}"
        kind = sub.source_kind.iloc[0]
        y0 = T - (r + 1) * row_h - r * ROW_GAP
        n = len(sub)
        for c, key in enumerate(KEYS):
            fi = int(round(key * (n - 1)))
            ax = fig.add_axes([x_right + c * (img_w + FRAME_GAP), y0, img_w, row_h])
            ax.imshow(imread(FRAMES / f"ep{ep}_front_f{fi}.jpg"), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == len(EPISODES) - 1:
                ax.set_xlabel(KEY_TITLES[c], labelpad=1.5)
        # row label above the frame sequence (centred over the frames), not on a vertical y label
        fig.text(x_right + frames_w / 2, y0 + row_h + 0.012, f"{klabel}: Milk on plate",
                 ha="center", va="bottom")
        t = sub.frame_index.to_numpy() / 30.0
        for c, (dk, dlabel) in enumerate(DIMS["delta"]):
            ax = fig.add_axes([x_traces + c * (trace_w + TRACE_GAP), y0, trace_w, row_h])
            if kind == "robot_3cam":
                ax.plot(t, 100 * smooth(sub[f"gt_{dk}"]), color=style.INK, linewidth=0.9)
            for lk, _ in LAMS:
                ax.plot(t, 100 * smooth(sub[f"{lk}_{a.probe}_{dk}"]), color=colors[lk], linewidth=0.9)
            ax.axhline(0, color=style.INK_MUTED, linewidth=0.4, zorder=0)
            ax.grid(color=style.GRID, linewidth=0.4)
            ax.set_xlim(t[0], t[-1])
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.tick_params(axis="y", labelrotation=90, pad=2)
            ax.tick_params(axis="x", pad=1.5)
            for lab in ax.get_yticklabels():
                lab.set_va("center")
            if r == 0:
                # body size, not axes.titlesize (larger in the science style; plot_probe_tsne_combo.py does the same)
                ax.set_title(dlabel.replace(" [m / 5 frames]", " [cm]"), pad=3, fontsize=plt.rcParams["font.size"])
            if r == len(EPISODES) - 1:
                ax.set_xlabel("time [s]", labelpad=1)
            else:
                ax.set_xticklabels([])

    # one legend band above both halves: bar colours = LAMs (shared with the trace colours), plus GT
    handles = [Patch(facecolor=combo.COLORS[m], edgecolor=style.MARKER_EDGE, linewidth=0.5, label=l) for m, l in LEGEND]
    handles.append(Line2D([0], [0], color=style.INK, linewidth=1.0, label="Ground-truth robot EE motion"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005),
               columnspacing=1.6, handletextpad=0.5)
    suffix = "_science" if a.style == "science" else ""
    save_cropped(fig, f"realworld_probe_decoded{'' if a.probe == 'ridge' else '_mlp'}{suffix}", plt)


def save_cropped(fig, name, plt, pad=0.01):
    """style.save, but with the top/bottom whitespace cut: the bounding box is the artists'
    tight box vertically and the full figure width horizontally (keeps \textwidth exact)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox = Bbox([[0.0, tb.y0 - pad], [fig.get_figwidth(), tb.y1 + pad]])
    style.FIG_DIR.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = style.FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches=bbox, **kw)
        print(f"  wrote {out.relative_to(style.FIG_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
