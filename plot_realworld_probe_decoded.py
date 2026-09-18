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
# crop of the 4:3 front frame (fractions of width / height): the left quarter is the aluminium frame
# and the top bare wall; a milder crop keeps the gripper at the right edge fully visible in start/end frames
CROP = (0.17, 1.0, 0.10, 1.0)  # x0, x1, y0, y1
CROP_ASPECT = ((CROP[1] - CROP[0]) * 4) / ((CROP[3] - CROP[2]) * 3)  # width / height of the cropped tile
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
    ax.set_ylabel("MLP $R^2$ per action dim.")
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
    h = w * 0.245
    fig = plt.figure(figsize=(w, h))
    # Explicit geometry (figure fractions) so that (i) every image box has the video's 4:3
    # aspect, (ii) the trace axes of a row have exactly the image height, and (iii) the two
    # rows together span the probe axes' vertical extent (top of row 0 = top of the bars,
    # bottom of row 1 = bottom of the bars).
    L, R, B, T = 0.06, 0.995, 0.135, 0.80
    SEP_GAP, TAG_W = 0.02, 0.018            # gap either side of the vertical rule; vertical Robot/Human tags
    ROW_GAP, FRAME_GAP, TRACE_GAP, BLOCK_GAP = 0.025, 0.006, 0.026, 0.032
    row_h = (T - B - ROW_GAP) / len(EPISODES)
    img_w = row_h * h * CROP_ASPECT / w        # cropped frame tile, in figure-width fractions
    trace_w = row_h * h / w                    # SQUARE trace panels: width = row height
    frames_w = len(KEYS) * img_w + (len(KEYS) - 1) * FRAME_GAP
    traces_w = 3 * trace_w + 2 * TRACE_GAP
    # the probe axes take whatever width is left once the right block is laid out
    probe_w = R - L - 2 * SEP_GAP - TAG_W - frames_w - BLOCK_GAP - traces_w
    assert probe_w > 0.2, f"probe panel too narrow ({probe_w:.3f}); reduce gaps or figure height ratio"
    ax_probe = fig.add_axes([L, B, probe_w, T - B])
    probe_panel(ax_probe, r2)
    x_sep = L + probe_w + SEP_GAP
    # separator over the plotting region plus the tick-label band, not the full figure height
    fig.add_artist(Line2D([x_sep, x_sep], [B - 0.065, T], transform=fig.transFigure,
                          color=style.INK_MUTED, linewidth=0.6))
    x_right = x_sep + SEP_GAP + TAG_W
    x_traces = x_right + frames_w + BLOCK_GAP

    for r, (ep, klabel) in enumerate(EPISODES):
        sub = df[df.episode_index == ep].sort_values("frame_index")
        assert len(sub), f"episode {ep} not in {TRACES.name}"
        kind = sub.source_kind.iloc[0]
        y0 = T - (r + 1) * row_h - r * ROW_GAP
        n = len(sub)
        for c, key in enumerate(KEYS):
            fi = int(round(key * (n - 1)))
            ax = fig.add_axes([x_right + c * (img_w + FRAME_GAP), y0, img_w, row_h])
            img = imread(FRAMES / f"ep{ep}_front_f{fi}.jpg")
            H, W = img.shape[:2]
            ax.imshow(img[int(CROP[2] * H):int(CROP[3] * H), int(CROP[0] * W):int(CROP[1] * W)], aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if r == len(EPISODES) - 1:
                ax.set_xlabel(KEY_TITLES[c], labelpad=1.5)
            if c == 0:
                ax.set_ylabel(klabel, labelpad=2)   # short vertical tag; the task is named once above
        if r == 0:
            fig.text(x_right + frames_w / 2, y0 + row_h + 0.02, "Milk on plate", ha="center", va="bottom")
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
                if c == 0:  # unit once, right after the last tick label of the first panel
                    ticks = [tk for tk in ax.get_xticks() if t[0] <= tk <= t[-1]]
                    rc = plt.rcParams
                    ax.annotate("t [s]", xy=(ticks[-1], 0), xycoords=("data", "axes fraction"),
                                xytext=(4.5, -(rc["xtick.major.size"] + 1.5)), textcoords="offset points",
                                ha="left", va="top", annotation_clip=False)
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
