#!/usr/bin/env python
"""Videos of the two human <-> robot transfer results (the actual movements).

  figures/xemb_latent_grid.mp4         one latent per column, applied to robot and human start
                                       frames (the LAM decoder's rollout, step by step); the top
                                       row plays the real demo moment each latent was taken from
  figures/xemb_realworld_transfer.mp4  the four robot / human pairs of plot_xemb_realworld_transfer.py
                                       as real footage, side by side, with their latents filling in
  figures/xemb_transfer_video.mp4      both, with title cards -- the one to show
  figures/xemb_headline_*.mp4          short headline clips: one latent from a real demo (left) driving a
                                       robot frame and a human frame (model output); and one real
                                       robot / human pair with the nearest latent
All videos play at half speed (labelled 0.5x).

Reuses the selections and loaders of extract_xemb_latent_grid.py / extract_xemb_realworld_transfer.py,
so the videos show exactly what the figures show. GPU + LAM stack (st-07: mg-latent env):

    ~/miniconda3/envs/mg-latent/bin/python experiments/make_xemb_videos.py
"""
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_xemb_latent_grid as G          # noqa: E402
import extract_xemb_realworld_transfer as T   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
FPS = 15
GRID_STEPS = 3          # decoder steps of 9 frames = 0.9 s
TILE = 200
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (26, 26, 26)
MUTED = (107, 107, 107)


def text(img, s, xy, scale=0.55, color=INK, thick=1, center=False):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    x, y = xy
    if center:
        x -= w // 2
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def rdbu(x):
    """Diverging blue-white-red for x in [-1, 1] (matplotlib RdBu_r anchors), returns uint8 RGB."""
    anchors = np.array([[5, 48, 97], [67, 147, 195], [247, 247, 247], [214, 96, 77], [103, 0, 31]], float)
    t = (np.clip(x, -1, 1) + 1) / 2 * (len(anchors) - 1)
    i = np.clip(np.floor(t).astype(int), 0, len(anchors) - 2)
    f = (t - i)[..., None]
    return (anchors[i] * (1 - f) + anchors[i + 1] * f).astype(np.uint8)


def title_card(lines, size, n):
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    y = size[1] // 2 - 20 * (len(lines) - 1)
    for k, (s, sc) in enumerate(lines):
        text(img, s, (size[0] // 2, y + 40 * k), scale=sc, thick=2 if k == 0 else 1, center=True)
    return [img] * n


def fit(frames, size):
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = min(size[0] / w, size[1] / h)
        r = cv2.resize(f, (int(w * s), int(h * s)))
        canvas = np.full((size[1], size[0], 3), 255, np.uint8)
        y0, x0 = (size[1] - r.shape[0]) // 2, (size[0] - r.shape[1]) // 2
        canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        out.append(canvas)
    return out


# ----------------------------------------------------------------------------- grid video
def grid_video(fr, pol, src):
    n_t = GRID_STEPS * 9                       # video frames per pass (real time, 30 fps source -> 15 fps here)
    srcs, motions = [], []
    for lab, e, t in G.SELECTED:
        f = fr.get(e, list(range(t, t + n_t + 1)))
        srcs.append([cv2.resize(x.permute(1, 2, 0).numpy(), (TILE, TILE)) for x in f])
        motions.append(G.motion_and_static(pol, G.clip(fr, e, t))[0])
    rows = []                                  # rows[r][c] = list of keyframes (step 0..GRID_STEPS)
    for _, e, t in G.ROWS:
        v = G.clip(fr, e, t)
        _, zs = G.motion_and_static(pol, v)
        x0 = v[:, :1, :, 0]
        start = cv2.resize(T.u8(x0[0, 0]), (TILE, TILE))
        cols = []
        for m in motions:
            keys, x = [start], x0
            for _ in range(GRID_STEPS):
                x = T.decode(pol, x, (zs + m)[None])
                keys.append(cv2.resize(T.u8(x[0, 0]), (TILE, TILE)))
            cols.append(keys)
        rows.append((start, cols))
    ncol = len(G.SELECTED) + 1
    pad, head, foot, lw = 6, 70, 34, 70
    Wd = lw + ncol * (TILE + pad)
    Hd = head + (len(G.ROWS) + 1) * (TILE + pad) + 18 + foot
    frames = []
    for i in range(0, n_t + 1):                # every 30 fps frame at 15 fps -> half speed
        step, frac = divmod(i / 9, 1)
        step = int(step)
        img = np.full((Hd, Wd, 3), 255, np.uint8)
        text(img, "One latent action, applied to robot and human frames", (Wd // 2, 26), 0.75, thick=2, center=True)
        text(img, f"t = {i / 30:.1f} s  (0.5x)", (Wd - 170, 26), 0.55, MUTED)
        y = head
        text(img, "source", (6, y + TILE // 2 - 8), 0.5, MUTED)
        text(img, "demo", (6, y + TILE // 2 + 12), 0.5, MUTED)
        for c, (lab, e, t) in enumerate(G.SELECTED, start=1):
            x = lw + c * (TILE + pad)
            img[y:y + TILE, x:x + TILE] = srcs[c - 1][min(i, n_t)]
            role = "robot" if src[e] == "robot_3cam" else "human"
            text(img, f"latent from {role}", (x + TILE // 2, y - 8), 0.48, MUTED, center=True)
        y += TILE + pad + 18
        cv2.line(img, (lw, y - 12), (Wd - 6, y - 12), (200, 200, 200), 1)
        for r, ((role, _, _), (start, cols)) in enumerate(zip(G.ROWS, rows)):
            yy = y + r * (TILE + pad)
            text(img, role.capitalize(), (6, yy + TILE // 2 + 6), 0.55)
            img[yy:yy + TILE, lw:lw + TILE] = start
            for c, keys in enumerate(cols, start=1):
                x = lw + c * (TILE + pad)
                a = keys[min(step, GRID_STEPS)]
                b = keys[min(step + 1, GRID_STEPS)]
                img[yy:yy + TILE, x:x + TILE] = (a * (1 - frac) + b * frac).astype(np.uint8)
        yy = y + len(G.ROWS) * (TILE + pad) + 20
        text(img, "start frame", (lw + TILE // 2, yy), 0.55, center=True)
        for c, (lab, _, _) in enumerate(G.SELECTED, start=1):
            text(img, lab, (lw + c * (TILE + pad) + TILE // 2, yy), 0.6, thick=2, center=True)
        frames.append(img)
    hold = [frames[-1]] * FPS
    return ([frames[0]] * (FPS // 2) + frames + hold) * 2   # loop twice


# ----------------------------------------------------------------------------- pairs video
def pairs_video(fr, stage):
    d, Z, st = T.load_latents(stage)
    _, mu = T.ridge_dirs(d, Z, st)
    MU = {"robot": mu[0], "human": mu[1]}
    key = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_arrays([d.episode_index.values, d.frame_index.values]))
    W, lead = T.W, 15
    VW, VH = 400, 300
    hm_h, hm_w = 64, 2 * VW
    frames = []
    allz = []
    for name, er, tr, eh, th in T.SELECTED:
        for role, e, t in (("robot", er, tr), ("human", eh, th)):
            allz.append(Z[[key[(e, t + i)] for i in range(W)]] - MU[role])
    vmax = np.abs(np.concatenate(allz)).max()
    for name, er, tr, eh, th in T.SELECTED:
        clips, zs = {}, {}
        for role, e, t in (("robot", er, tr), ("human", eh, th)):
            f = fr.get(e, list(range(t - lead, t + W + lead)))
            clips[role] = [cv2.resize(x.permute(1, 2, 0).numpy(), (VW, VH)) for x in f]
            zs[role] = Z[[key[(e, t + i)] for i in range(W)]] - MU[role]
        n = len(clips["robot"])
        for i in range(0, n):                  # half speed
            img = np.full((60 + VH + 30 + 2 * hm_h + 50, 2 * VW + 30, 3), 255, np.uint8)
            text(img, f"movement: {name.replace('_', '-')}", (img.shape[1] // 2, 30), 0.8, thick=2, center=True)
            text(img, "0.5x", (img.shape[1] - 50, 30), 0.5, MUTED)
            inwin = lead <= i < lead + W
            for k, role in enumerate(("robot", "human")):
                x = 10 + k * (VW + 10)
                img[60:60 + VH, x:x + VW] = clips[role][i]
                if inwin:
                    cv2.rectangle(img, (x, 60), (x + VW - 1, 60 + VH - 1), (0, 114, 178), 4)
                text(img, role.capitalize(), (x + 8, 84), 0.7, (255, 255, 255), 3)
                text(img, role.capitalize(), (x + 8, 84), 0.7, INK, 1)
            y0 = 60 + VH + 30
            text(img, "latent (8 dims) over the 1 s window, minus each embodiment's average", (10, y0 - 8), 0.5, MUTED)
            upto = int(np.clip(i - lead + 1, 0, W))
            for k, role in enumerate(("robot", "human")):
                hm = np.full((hm_h, hm_w, 3), 245, np.uint8)
                if upto:
                    col = rdbu(zs[role][:upto].T / vmax)                      # (8, upto, 3)
                    col = cv2.resize(col, (int(hm_w * upto / W), hm_h), interpolation=cv2.INTER_NEAREST)
                    hm[:, :col.shape[1]] = col
                yy = y0 + k * (hm_h + 4)
                img[yy:yy + hm_h, 10 + 20:10 + 20 + hm_w - 20] = hm[:, :hm_w - 20]
                text(img, role[0].upper(), (10, yy + hm_h // 2 + 6), 0.55)
            frames.append(img)
        frames += [frames[-1]] * FPS
    return frames


HEADLINES = [   # (file stem, latent label in G.SELECTED, robot start-frame row, human start-frame row)
    ("xemb_headline_away", "away", 0, 2),
    ("xemb_headline_up_left", "up-left", 0, 3),
    ("xemb_headline_left", "left", 1, 2),
]
BIG = 360


def headline_latent(fr, pol, src, stem, label, r_rob, r_hum):
    lab, e, t = next(x for x in G.SELECTED if x[0] == label)
    role = "robot" if src[e] == "robot_3cam" else "human"
    n_t = GRID_STEPS * 9
    srcf = [cv2.resize(x.permute(1, 2, 0).numpy(), (BIG, BIG)) for x in fr.get(e, list(range(t, t + n_t + 1)))]
    m = G.motion_and_static(pol, G.clip(fr, e, t))[0]
    keys = []
    for ri in (r_rob, r_hum):
        _, er, tr = G.ROWS[ri]
        v = G.clip(fr, er, tr)
        _, zs = G.motion_and_static(pol, v)
        x = v[:, :1, :, 0]
        k = [cv2.resize(T.u8(x[0, 0]), (BIG, BIG))]
        for _ in range(GRID_STEPS):
            x = T.decode(pol, x, (zs + m)[None])
            k.append(cv2.resize(T.u8(x[0, 0]), (BIG, BIG)))
        keys.append(k)
    pad, head, foot = 16, 96, 46
    Wd, Hd = 3 * BIG + 4 * pad + 30, head + BIG + foot
    frames = []
    for i in range(n_t + 1):
        step, frac = divmod(i / 9, 1)
        step = int(step)
        img = np.full((Hd, Wd, 3), 255, np.uint8)
        text(img, f"One latent action ({label}), taken from a {role} demo", (Wd // 2, 36), 0.85, thick=2, center=True)
        text(img, "applied by the latent action model to a robot frame and a human frame", (Wd // 2, 68), 0.6, MUTED,
             center=True)
        x0 = pad
        img[head:head + BIG, x0:x0 + BIG] = srcf[i]
        text(img, f"real {role} demo (latent source)", (x0 + BIG // 2, head + BIG + 30), 0.55, center=True)
        cv2.arrowedLine(img, (x0 + BIG + 6, head + BIG // 2), (x0 + BIG + pad + 24, head + BIG // 2), INK, 2, tipLength=0.4)
        for k, (who, kk) in enumerate((("robot", keys[0]), ("human", keys[1]))):
            x = pad + (k + 1) * (BIG + pad) + 30
            a, b = kk[min(step, GRID_STEPS)], kk[min(step + 1, GRID_STEPS)]
            img[head:head + BIG, x:x + BIG] = (a * (1 - frac) + b * frac).astype(np.uint8)
            text(img, f"{who} frame + latent (model)", (x + BIG // 2, head + BIG + 30), 0.55, center=True)
        text(img, "0.5x", (Wd - 50, 36), 0.5, MUTED)
        frames.append(img)
    clip = [frames[0]] * (FPS // 2) + frames + [frames[-1]] * FPS
    imageio.mimsave(FIG / f"{stem}.mp4", clip * 3, fps=FPS, quality=8, macro_block_size=8)
    return FIG / f"{stem}.mp4"


def headline_pair(fr, name, out_stem):
    _, er, tr, eh, th = next(x for x in T.SELECTED if x[0] == name)
    lead, W = 15, T.W
    clips = [[cv2.resize(x.permute(1, 2, 0).numpy(), (480, 360)) for x in fr.get(e, list(range(t - lead, t + W + lead)))]
             for e, t in ((er, tr), (eh, th))]
    pad, head, foot = 16, 96, 46
    Wd, Hd = 2 * 480 + 3 * pad, head + 360 + foot
    frames = []
    for i in range(len(clips[0])):
        img = np.full((Hd, Wd, 3), 255, np.uint8)
        text(img, "Robot and human clips with the nearest latent", (Wd // 2, 36), 0.85, thick=2, center=True)
        text(img, f"same latent -> same movement ({name.replace('_', '-')})", (Wd // 2, 68), 0.6, MUTED, center=True)
        for k, who in enumerate(("robot", "human")):
            x = pad + k * (480 + pad)
            img[head:head + 360, x:x + 480] = clips[k][i]
            if lead <= i < lead + W:
                cv2.rectangle(img, (x, head), (x + 479, head + 359), (0, 114, 178), 4)
            text(img, f"real {who} demo", (x + 240, head + 360 + 30), 0.6, center=True)
        text(img, "0.5x", (Wd - 50, 36), 0.5, MUTED)
        frames.append(img)
    imageio.mimsave(FIG / f"{out_stem}.mp4", (frames + [frames[-1]] * FPS) * 3, fps=FPS, quality=8, macro_block_size=8)
    return FIG / f"{out_stem}.mp4"


def main():
    stage = "/home/admin_07/.claude/jobs/7ac23333/tmp/dk1"
    root = str(Path.home() / "rlfv_stage/dk1-postprocessed-full-3cam-placeholder-20260719")
    ckpt = "/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam2_43524549_030000"
    lam_src = "/home/admin_07/project_repos/lerobot_policy_lam_plain_dino/src"
    fr = T.Frames(root)
    pol = T.load_policy(ckpt, lam_src)
    with torch.inference_mode():
        import types
        src, _ = G.episode_meta(types.SimpleNamespace(stage=stage))
        gv = grid_video(fr, pol, src)
        heads = [headline_latent(fr, pol, src, *h) for h in HEADLINES]
    heads.append(headline_pair(fr, "up_right", "xemb_headline_pair_banana"))
    pv = pairs_video(fr, stage)
    kw = dict(fps=FPS, quality=8, macro_block_size=8)
    imageio.mimsave(FIG / "xemb_latent_grid.mp4", gv, **kw)
    imageio.mimsave(FIG / "xemb_realworld_transfer.mp4", pv, **kw)
    size = (max(gv[0].shape[1], pv[0].shape[1]), max(gv[0].shape[0], pv[0].shape[0]))
    allf = title_card([("Same latent, same movement: robot and human", 0.9),
                       ("real-world DK1, multi-view latent action model", 0.6)], size, FPS * 3)
    allf += title_card([("1. Robot and human clips with the nearest latent", 0.8),
                        ("same movement -> same latent; different movement -> different latent", 0.55)], size, FPS * 3)
    allf += fit(pv, size)
    allf += title_card([("2. One latent action, applied to robot and human frames", 0.8),
                        ("each column: one latent from one real demo moment; the model's decoder", 0.55),
                        ("moves the robot gripper and the human hand the same way", 0.55)], size, FPS * 4)
    allf += fit(gv, size)
    imageio.mimsave(FIG / "xemb_transfer_video.mp4", allf, **kw)
    for n in ("xemb_latent_grid.mp4", "xemb_realworld_transfer.mp4", "xemb_transfer_video.mp4"):
        print("wrote", FIG / n)
    for hpath in heads:
        print("wrote", hpath)


if __name__ == "__main__":
    main()
