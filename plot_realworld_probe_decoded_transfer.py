r"""Variant of plot_realworld_probe_decoded.py: the left probe bars are replaced by a human -> robot
latent-transfer example (milk on pink plate, moving right; example 0 of results/xemb_traj_h2r_lr.csv,
human ep 1626 @ 100 -> robot ep 597 @ 130, frames from results/frames_xemb_traj_h2r_lr/).
Top row: the human demo at frame [0] and frame [FRAME_B] (arrow in between = elapsed time); the right
frame carries the movement-direction arrow. Bottom row: the robot start frame and the LAM decoder's
prediction after the human's latents were applied (raw model output). Right block: unchanged.
Emits figures/realworld_probe_decoded_transfer[_science].{pdf,png}.

Original description follows.

Real-world probing + decoded human-video motion in ONE double-column figure.

Combines the two real-world figures into a single ``figure*`` (IEEE double column,
height:width about 1:5):

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
# 2026-10-03 (user): every trace in the right block starts at 0 at t=0, i.e. each smoothed curve
# (ground truth and both decodings) is shifted by its own first value. Presentation only; the
# CSV traces are untouched.
ZERO_START = True


def trace(v):
    """Smoothed trace in cm; with ZERO_START the curve is offset so that it starts at 0."""
    y = 100 * smooth(v)
    return y - y[0] if ZERO_START else y


LINESTYLE = {"ours_multi": "-", "ours_single": "--"}
EPISODES = [(1638, "Human"), (596, "Robot")]   # right block: human on top, robot below (overrides the original order)
TRANSFER_CSV = HERE / "results" / "xemb_traj_h2r_lr.csv"
TRANSFER_FRAMES = HERE / "results" / "frames_xemb_traj_h2r_lr"
TRANSFER_EX = 0          # milk on pink plate, moving right
FRAME_B = 4              # second frame index; frames are 0.3 s apart -> FRAME_B * 0.3 s
ARROW_GAIN = 2.5         # image arrows = the human hand's measured motion over the interval, drawn at 2.5x
# the same arrow (vector and TIP position) is drawn in both "after" frames, so it reads as one movement shown on
# two embodiments. The arrow comes in from the left and its head stops at the tip of the hand / end effector:
# in example 0 the human's fingertips on the milk carton and the left edge of the decoded gripper both sit at
# about x 0.69-0.70, y 0.60 (read off a coordinate grid over the two frames).
ARROW_TIP = (0.695, 0.60)   # (x, y) image fractions, y down: where the arrowhead ends


def movement_arrow(ax, tx, ty, u, v):
    """Arrow of image-fraction vector (u, v) whose HEAD ends at (tx, ty) (image coordinates, y down)."""
    import matplotlib.patheffects as pe
    a = ax.annotate("", xy=(tx, 1 - ty), xytext=(tx - u, 1 - (ty - v)),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.18",
                                                              color=style.INK, linewidth=1.1, shrinkA=0, shrinkB=0))
    a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.4, foreground="white"), pe.Normal()])


def transfer_panel(fig, x0, width, T, row_h, ROW_GAP, w, h, plt, imread):
    """Two rows (human, robot) x [frame 0 -> frame FRAME_B], aligned with the right block's rows."""
    from matplotlib.patches import FancyArrowPatch
    r = pd.read_csv(TRANSFER_CSV).set_index("example").loc[TRANSFER_EX]
    tile_w = row_h * h * (4 / 3) / w          # 4:3 frame tiles, same height as the right block's rows
    gap = width - 2 * tile_w                   # room for the time arrow between the two frames
    assert gap > 0.02, f"transfer panel too narrow ({gap:.3f})"
    rows = [("human", "human", "Human"), ("robot_model", "robot_model", "Robot")]
    arrow_c = ARROW_TIP
    fs = plt.rcParams["font.size"]
    for ri, (tag0, tagb, label) in enumerate(rows):
        y0 = T - (ri + 1) * row_h - ri * ROW_GAP
        for c, (tag, k) in enumerate(((tag0, 0), (tagb, FRAME_B))):
            ax = fig.add_axes([x0 + c * (tile_w + gap), y0, tile_w, row_h])
            ax.imshow(imread(TRANSFER_FRAMES / f"ex{TRANSFER_EX}_{tag}_k{k}.png"), aspect="auto", extent=(0, 1, 0, 1))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if c == 0:
                ax.set_ylabel(label, labelpad=2)
            if c == 1:
                movement_arrow(ax, arrow_c[0], arrow_c[1], ARROW_GAIN * r.human_u, ARROW_GAIN * r.human_v)
                if ri == 1:
                    ax.text(0.5, 0.95, "+ human latents", transform=ax.transAxes, ha="center", va="top",
                            fontsize=fs - 2.5, color=style.INK)
            if ri == 0 and c == 0:
                ax.text(0.5, 0.95, "Milk on plate", transform=ax.transAxes, ha="center", va="top",
                        fontsize=fs - 1.5, color=style.INK)
        # left-to-right arrow between the two frames of each row; the elapsed time is written in the human row only
        ym = y0 + row_h / 2
        xa, xb = x0 + tile_w + 0.006, x0 + tile_w + gap - 0.006
        fig.add_artist(FancyArrowPatch((xa, ym), (xb, ym), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=7, color=style.INK, linewidth=0.8))
        if ri == 0:
            fig.text((xa + xb) / 2, ym + 0.02, f"{FRAME_B * 0.3:.1f} s", ha="center", va="bottom", fontsize=fs - 1.5)
    # human latents -> robot: vertical arrow in the gap column, from below the time arrow down to the robot row
    xm = x0 + tile_w + gap / 2
    y_top = T - row_h / 2 - 0.05                 # just under the human row's time arrow
    y_bot = T - 1.5 * row_h - ROW_GAP + 0.035   # stops just above the robot row's horizontal arrow
    fig.add_artist(FancyArrowPatch((xm, y_top), (xm, y_bot), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=7, color=style.INK, linewidth=0.8))
    fig.text(xm - 0.004, (y_top + y_bot) / 2, "transfer latent", ha="right", va="center", rotation=90,
             fontsize=fs - 2.5)
    print(f"transfer example: human ep {int(r.human_episode)} @ {int(r.human_frame)} -> robot ep "
          f"{int(r.robot_episode)} @ {int(r.robot_frame)} ({r.task_human}), frames 0 and {FRAME_B} "
          f"({FRAME_B * 0.3:.1f} s); cos vs human {r.cos_vs_human:.2f}, vs robot real {r.cos_vs_robot_real:.2f}")


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
    ax.set_yticks(np.arange(0, 1.01, 0.2))   # fixed, so a shorter axis cannot fall back to 0.25 steps (wider labels)
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
    # Vertical geometry in inches: the plotting region (two image rows == probe bars) sits
    # between a bottom band (tick labels, "start / 50 % / end") and a top band that holds
    # only the legend -- the trace panels are labelled on their y axes, and the task name
    # is written into the first key frame, so nothing else needs headroom.
    BOT_IN, ROWS_IN, TOP_IN = 0.236, 1.08, 0.24
    h = BOT_IN + ROWS_IN + TOP_IN
    fig = plt.figure(figsize=(w, h))
    # Explicit geometry (figure fractions) so that (i) every image box has the video's 4:3
    # aspect, (ii) the trace axes of a row have exactly the image height, and (iii) the two
    # rows together span the probe axes' vertical extent (top of row 0 = top of the bars,
    # bottom of row 1 = bottom of the bars).
    L, R, B, T = 0.058, 0.995, BOT_IN / h, (BOT_IN + ROWS_IN) / h
    SEP_GAP, TAG_W = 0.012, 0.018            # gap either side of the vertical rule; vertical Robot/Human tags
    ROW_GAP, FRAME_GAP = 0.025, 0.006
    TRACE_GAP, BLOCK_GAP = 0.046, 0.052      # each holds a trace panel's rotated tick labels plus its y label
    row_h = (T - B - ROW_GAP) / len(EPISODES)
    img_w = row_h * h * CROP_ASPECT / w        # cropped frame tile, in figure-width fractions
    trace_w = row_h * h / w                    # SQUARE trace panels: width = row height
    frames_w = len(KEYS) * img_w + (len(KEYS) - 1) * FRAME_GAP
    traces_w = 3 * trace_w + 2 * TRACE_GAP
    # the probe axes take whatever width is left once the right block is laid out
    probe_w = R - L - 2 * SEP_GAP - TAG_W - frames_w - BLOCK_GAP - traces_w
    assert probe_w > 0.2, f"probe panel too narrow ({probe_w:.3f}); reduce gaps or figure height ratio"
    transfer_panel(fig, L, probe_w, T, (T - B - ROW_GAP) / len(EPISODES), ROW_GAP, w, h, plt, imread)
    x_sep = L + probe_w + SEP_GAP
    # separator over the plotting region plus the tick-label band, not the full figure height
    fig.add_artist(Line2D([x_sep, x_sep], [B - 0.065, T], transform=fig.transFigure,
                          color=style.INK_MUTED, linewidth=0.6))
    x_right = x_sep + SEP_GAP + TAG_W
    x_traces = x_right + frames_w + BLOCK_GAP

    legend_lines = {}
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
                ax.set_ylabel(klabel, labelpad=2)   # short vertical tag; the task is named once, in a frame
            if r == 0 and c == 1:
                # task name inside the frame (top centre, small, straight on the bare wall) instead of a
                # text band above the row, which is what keeps the figure this short
                ax.text(0.5, 0.95, "Milk on plate", transform=ax.transAxes, ha="center", va="top",
                        fontsize=plt.rcParams["font.size"] - 1.5, color=style.INK)
        t = sub.frame_index.to_numpy() / 30.0
        for c, (dk, dlabel) in enumerate(DIMS["delta"]):
            ax = fig.add_axes([x_traces + c * (trace_w + TRACE_GAP), y0, trace_w, row_h])
            if kind == "robot_3cam":
                ax.plot(t, trace(sub[f"gt_{dk}"]), color=style.INK, linewidth=0.9, linestyle="-")
            for lk, _ in LAMS:
                # explicit styles: the same LAM looks the same in both rows (the style cycler would otherwise
                # shift by one in the robot row, where the ground-truth line is drawn first)
                (ln,) = ax.plot(t, trace(sub[f"{lk}_{a.probe}_{dk}"]), color=colors[lk], linewidth=0.9,
                                linestyle=LINESTYLE[lk])
                legend_lines[lk] = ln
            ax.axhline(0, color=style.INK_MUTED, linewidth=0.4, zorder=0)
            ax.grid(color=style.GRID, linewidth=0.4)
            ax.set_xlim(t[0], t[-1])
            ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.tick_params(axis="y", labelrotation=90, pad=2)
            ax.tick_params(axis="x", pad=1.5)
            for lab in ax.get_yticklabels():
                lab.set_va("center")
            # dimension + unit on the y axis of every panel (each has its own scale), no title band
            ax.set_ylabel(dlabel.replace(" [m / 5 frames]", " [cm]"), labelpad=2)
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
    handles = [Line2D([0], [0], color=combo.COLORS[m], linewidth=1.0, linestyle=legend_lines[m].get_linestyle(), label=l)
               for m, l in LEGEND]
    handles.append(Line2D([0], [0], color=style.INK, linewidth=1.0, label="Ground-truth robot EE motion"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=((x_sep + R) / 2, 1.005),
               columnspacing=1.6, handletextpad=0.5)
    suffix = "_science" if a.style == "science" else ""
    save_cropped(fig, f"realworld_probe_decoded_transfer{'' if a.probe == 'ridge' else '_mlp'}{suffix}", plt)


def save_cropped(fig, name, plt, pad=0.01):
    """style.save, but with the top/bottom whitespace cut: the bounding box is the artists'
    tight box vertically and the full figure width horizontally (keeps \textwidth exact)."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    bbox = Bbox([[tb.x0 - pad, tb.y0 - pad], [tb.x1 + pad, tb.y1 + pad]])   # tight on all sides (user request)
    style.FIG_DIR.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        out = style.FIG_DIR / f"{name}.{ext}"
        fig.savefig(out, bbox_inches=bbox, **kw)
        print(f"  wrote {out.relative_to(style.FIG_DIR.parent)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
