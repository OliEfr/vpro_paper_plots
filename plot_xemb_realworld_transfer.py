r"""Same latent, same movement: human <-> robot latent transfer on real-world DK1 data.

Reads the dumps written by ``experiments/extract_xemb_realworld_transfer.py``:
``results/xemb_realworld_transfer.csv`` (the 30x8 latent sequences),
``results/xemb_realworld_transfer_arrows.csv`` (motion arrows) and the key frames in
``results/frames_xemb_realworld/``. Emits ``figures/xemb_realworld_transfer[_science].{pdf,png}``.

What is shown. Multi-view LAM (front + side, the paper's teacher), raw 8-D latents, no
normalisation of any kind.

  (a) A 1 s ROBOT clip (top) and the HUMAN clip whose latent sequence is its nearest
      neighbour among all human windows (bottom), same task "close drawer": three front
      key frames each (start, 0.5 s, 1 s), the image motion of the hand / gripper over the
      clip as an arrow (optical flow, top-3 % magnitude pixels), and the two latent
      sequences as heatmaps on one colour scale (rows = latent dims, columns = frames).
  (b) The same for a cross-task pair: the robot closing a drawer and a human pushing a
      bowl away -- different object and task, same latent, same pushing movement.
  Arrows are drawn at 2x their measured length (ARROW_GAIN).
  (c) The latent as a cause rather than a correlate: one latent edit (the direction a
      ridge probe fitted on ROBOT data reads as end-effector +y / +z) is added to the
      mean latent and decoded by the LAM's own pixel decoder on a robot frame and on a
      human frame; arrows = image motion between the -6 cm and +6 cm decodings. The same
      edit moves the gripper and the hand the same way.

The examples are cherry-picked by design (the question is whether a clear example
exists); the stats csv carries how often matched pairs agree vs random pairs.

LaTeX:

    \begin{figure*}[t]
      \centering
      \includegraphics[width=\textwidth]{figures/xemb_realworld_transfer.pdf}
      \caption{Same latent, same movement across embodiments (real world, multi-view
        LAM, raw latents). (a)~A robot clip and the human clip with the nearest latent
        sequence both push the drawer shut; arrows: image motion over 1\,s; heatmaps:
        the two 8-D latent sequences on one colour scale; arrows drawn at $2\times$ length. (b)~Across tasks: the robot
        closing a drawer and a human pushing a bowl away share the latent and the
        movement. (c)~Decoding one latent edit (robot-probe $+y$, $+z$) on a robot and a
        human frame moves gripper and hand alike.}
      \label{fig:xemb_realworld_transfer}
    \end{figure*}

Usage:
    python plot_xemb_realworld_transfer.py [--style paper|science]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
LAT = HERE / "results" / "xemb_realworld_transfer.csv"
ARR = HERE / "results" / "xemb_realworld_transfer_arrows.csv"
FRAMES = HERE / "results" / "frames_xemb_realworld"
PAIRS = [("close_drawer", "(a) same task: close drawer"),
         ("push_away", "(b) across tasks: close drawer / push bowl away")]
ROLES = [("robot", "Robot"), ("human", "Human")]
KEY_TITLES = ["0 s", "0.5 s", "1 s"]
CROP_FRAC = 0.58          # crop side, fraction of frame width / height, centred on the moving hand / gripper
ARROW_GAIN = 2.0          # arrows drawn at 2x their measured length, for legibility at print size
EDITS = [("edit_y", r"latent edit: robot-probe $+y$"), ("edit_z", r"latent edit: robot-probe $+z$")]


def crop_box(cx, cy):
    """Crop (fractions) centred on the arrow, clipped to the frame."""
    h = CROP_FRAC / 2
    x0 = float(np.clip(cx - h, 0, 1 - CROP_FRAC))
    y0 = float(np.clip(cy - h, 0, 1 - CROP_FRAC))
    return x0, y0


def show(ax, img, box):
    H, Wd = img.shape[:2]
    x0, y0 = box
    ax.imshow(img[int(y0 * H):int((y0 + CROP_FRAC) * H), int(x0 * Wd):int((x0 + CROP_FRAC) * Wd)],
              aspect="auto", extent=(x0, x0 + CROP_FRAC, y0 + CROP_FRAC, y0))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def arrow(ax, r, color, ls="-", plt=None):
    import matplotlib.patheffects as pe
    u, v = ARROW_GAIN * r.u, ARROW_GAIN * r.v
    x, y = r.cx - u / 2, r.cy - v / 2
    a = ax.annotate("", xy=(x + u, y + v), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.25", color=color,
                                    linewidth=1.4, linestyle=ls, shrinkA=0, shrinkB=0))
    a.arrow_patch.set_path_effects([pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()])
    a.arrow_patch.set_clip_path(ax.patch)   # keep the arrow inside its tile


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
    from matplotlib.lines import Line2D

    lat = pd.read_csv(LAT)
    arr = pd.read_csv(ARR)
    zc = [f"z{k}" for k in range(8)]

    w = style.TEXT_WIDTH
    tile_w = 0.70                     # inch; crop is CROP_FRAC of a 4:3 frame -> tile aspect 4:3
    tile_h = tile_w * 3 / 4
    heat_w = 0.80
    gap_pair = 0.32
    top, bottom, row_gap, block_gap = 0.30, 0.10, 0.04, 0.34
    h = top + 2 * tile_h + row_gap + block_gap + tile_h + 0.12 + bottom
    fig = plt.figure(figsize=(w, h))
    X = lambda inch: inch / w
    Y = lambda inch: 1 - inch / h     # from the top

    left0 = 0.30
    cmap = plt.get_cmap("RdBu_r")
    print(f"{'pair':14s} {'role':6s} episode frame  task")
    for pi, (pair, ptitle) in enumerate(PAIRS):
        x_start = left0 + pi * (3 * tile_w + 0.17 + heat_w + gap_pair)
        sub = lat[lat.pair == pair]
        vmax = np.abs(sub[zc].values).max()
        fig.text(X(x_start), Y(0.10), ptitle, ha="left", va="top", fontsize=plt.rcParams["font.size"])
        for ri, (role, rlabel) in enumerate(ROLES):
            y_top = top + ri * (tile_h + row_gap)
            r_arr = arr[(arr.panel == pair) & (arr.role == role) & (arr.kind == "window")].iloc[0]
            box = crop_box(r_arr.cx, r_arr.cy)
            for j in range(3):
                ax = fig.add_axes([X(x_start + j * tile_w), Y(y_top + tile_h), X(tile_w - 0.02), tile_h / h])
                show(ax, imread(FRAMES / f"{pair}_{role}_k{j}.jpg"), box)
                if j == 2:
                    arrow(ax, r_arr, style.INK)
                if ri == 0:
                    ax.set_title(KEY_TITLES[j], pad=1.5, fontsize=plt.rcParams["font.size"] - 1)
                if j == 0 and pi == 0:
                    ax.set_ylabel(rlabel, labelpad=2)
            zz = sub[sub.role == role].sort_values("step")
            axh = fig.add_axes([X(x_start + 3 * tile_w + 0.17), Y(y_top + tile_h), X(heat_w), tile_h / h])
            im = axh.imshow(zz[zc].values.T, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
            axh.set_yticks([0, 7]); axh.set_yticklabels(["$z_1$", "$z_8$"])
            axh.minorticks_off()
            axh.tick_params(which="both", length=1.5, pad=1, top=False, right=False)
            if ri == 0:
                axh.set_xticks([]); axh.set_title("latent (8-D) over 1 s", pad=1.5, fontsize=plt.rcParams["font.size"] - 1)
            else:
                axh.set_xticks([0, 29]); axh.set_xticklabels(["0 s", "1 s"])
            for s in axh.spines.values():
                s.set_visible(False)
            e0 = zz.iloc[0]
            print(f"{pair:14s} {role:6s} {int(e0.episode):7d} {int(e0.frame):5d}  {e0.task}")

    # (c) traversal
    y_top = top + 2 * tile_h + row_gap + block_gap
    fig.text(X(left0), Y(y_top - 0.06), "(c) one latent edit, decoded on a robot and a human frame",
             ha="left", va="bottom", fontsize=plt.rcParams["font.size"])
    colors = {"edit_y": style.PALETTE[0], "edit_z": style.PALETTE[2]}
    dashes = {"edit_y": "-", "edit_z": (0, (2.2, 1.2))}
    for ri, (role, rlabel) in enumerate(ROLES):
        win = arr[(arr.panel == "close_drawer") & (arr.role == role) & (arr.kind == "window")].iloc[0]
        box = crop_box(win.cx, win.cy)
        ax = fig.add_axes([X(left0 + ri * (tile_w + 0.06)), Y(y_top + tile_h), X(tile_w - 0.02), tile_h / h])
        show(ax, imread(FRAMES / f"close_drawer_{role}_k0.jpg"), box)
        ax.set_xlabel(rlabel, labelpad=1.5)
        for kind, _ in EDITS:
            r = arr[(arr.panel == "traverse") & (arr.role == role) & (arr.kind == kind)].iloc[0]
            arrow(ax, r, colors[kind], dashes[kind])
    handles = [Line2D([0], [0], color=colors[k], linestyle=dashes[k], linewidth=1.4, label=l) for k, l in EDITS]
    handles.append(Line2D([0], [0], color=style.INK, linewidth=1.4, label="image motion of the clip, (a)/(b)"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(X(left0 + 2 * tile_w + 0.25), Y(y_top + tile_h / 2)),
               frameon=False, handlelength=2.2)
    cax = fig.add_axes([X(w - 1.55), Y(y_top + tile_h * 0.62), X(1.2), 0.07 / h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([]); cb.outline.set_visible(False)
    cax.set_title("latent value: $-$  0  $+$ (one scale per pair)", pad=1.5, fontsize=plt.rcParams["font.size"] - 1)

    print("\narrows (fractions of frame):")
    print(arr.to_string(index=False))
    suffix = "_science" if a.style == "science" else ""
    style.save(fig, f"xemb_realworld_transfer{suffix}")


if __name__ == "__main__":
    main()
