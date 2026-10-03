#!/usr/bin/env python
"""Additional human <-> robot transfer results (new files only; the earlier results are untouched).

1. SHARP latent grid. Same latents / start frames as extract_xemb_latent_grid.py, but instead of the
   LAM decoder's (blurry, 256 px, 32 px patches) image, the REAL full-resolution start frame is warped
   by the motion the decoder predicts: optical flow between consecutive decoder outputs, chained over
   3 steps (0.9 s). Sharp, but a visualisation: large motions stretch the hand / arm ("rubber sheet").
   -> results/frames_xemb_latent_grid_sharp/r{row}_c{col}.png, figures/xemb_latent_grid_sharp.mp4

2. TRAJECTORY transfer. The latent SEQUENCE of a real clip -- one motion latent per 9 frames, i.e. every
   0.3 s, each = clip latent minus the same frame held still -- drives a start frame of the other
   embodiment for K=4 steps (1.2 s), both camera views decoded so the "held still" reference is
   recomputed from the model's own prediction at every step. Rendered both as decoder output and as the
   sharp chained warp. Scored by the optical-flow direction of the generated motion (vs a zero-motion
   rollout) against (a) the source clip's real motion and (b) the target's own real future (the pairs
   were chosen so both embodiments really did the same movement).
   -> results/xemb_trajectory_transfer.csv (scores), results/frames_xemb_trajectory/*.png,
      figures/xemb_trajectory_transfer.mp4, figures/xemb_headline_trajectory_*.mp4

GPU + LAM stack (st-07: mg-latent env):
    ~/miniconda3/envs/mg-latent/bin/python experiments/extract_xemb_additional.py
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
from make_xemb_videos import INK, MUTED, text  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES, FIG = HERE / "results", HERE / "figures"
GRID_DIR = RES / "frames_xemb_latent_grid_sharp"
TRAJ_DIR = RES / "frames_xemb_trajectory"
H, WD = 480, 640
STEPS_GRID, K, S = 3, 4, 9
FPS = 15
# (name, robot episode, robot frame, human episode, human frame): both embodiments really make this movement
PAIRS = [("push_cup_right", 272, 140, 1258, 195), ("banana_lift_up_right", 486, 115, 790, 95),
         ("close_drawer_away", 112, 435, 997, 275), ("down_right", 200, 140, 1645, 105)]
YY, XX = np.mgrid[0:H, 0:WD].astype(np.float32)


def flow_back(prev, cur):
    """Flow from cur's pixel grid back to prev (for backward warping), upsampled to full resolution."""
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(g(cur), g(prev), None, 0.5, 4, 21, 5, 7, 1.5, 0)
    f = cv2.resize(f, (WD, H))
    f[..., 0] *= WD / 256
    f[..., 1] *= H / 256
    return f


def warp(img, f):
    return cv2.remap(img, XX + f[..., 0], YY + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def real_full(fr, e, t):
    return fr.get(e, [t])[0].permute(1, 2, 0).numpy()


@torch.inference_mode()
def static_lat(pol, first):
    return G.lat(pol, first[:, :, :, None].repeat(1, 1, 1, 4, 1, 1))[0]


@torch.inference_mode()
def drive(pol, fr, e, t, motions):
    """Roll the LAM decoder from (e, t) under a list of motion latents; returns decoder frames (front, 256 px)
    and the chained sharp warp of the real full-res frame."""
    v = G.clip(fr, e, t)
    x = v[:, :, :, 0]                                   # (1, V, 3, H, W): both views
    prev = T.u8(T.decode(pol, x, static_lat(pol, x)[None])[0, 0])   # "held still" prediction
    sharp = real_full(fr, e, t)
    dec, shp = [T.u8(x[0, 0])], [sharp]
    for m in motions:
        x = T.decode(pol, x, (static_lat(pol, x) + m)[None])
        cur = T.u8(x[0, 0])
        sharp = warp(sharp, flow_back(prev, cur))
        prev = cur
        dec.append(cur)
        shp.append(sharp)
    return dec, shp


def motion_seq(pol, fr, e, t):
    return [G.motion_and_static(pol, G.clip(fr, e, t + S * k))[0] for k in range(K)]


def real_seq(fr, e, t, size=256):
    f = torch.nn.functional.interpolate(fr.get(e, [t + S * k for k in range(K + 1)]).float().cuda() / 255.0,
                                        size=(size, size), mode="bilinear")
    return [T.u8(x) for x in f]


def sharp_grid(fr, pol, src):
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    motions = [G.motion_and_static(pol, G.clip(fr, e, t))[0] for _, e, t in G.SELECTED]
    cells = {}
    for ri, (_, e, t) in enumerate(G.ROWS):
        cv2.imwrite(str(GRID_DIR / f"r{ri}_c0.png"), cv2.cvtColor(real_full(fr, e, t), cv2.COLOR_RGB2BGR))
        for ci, m in enumerate(motions, start=1):
            _, shp = drive(pol, fr, e, t, [m] * STEPS_GRID)
            cells[(ri, ci)] = shp
            cv2.imwrite(str(GRID_DIR / f"r{ri}_c{ci}.png"), cv2.cvtColor(shp[-1], cv2.COLOR_RGB2BGR))
    # video: crossfade between steps, half speed
    T_ = 200
    ncol, pad, head, lw = len(G.SELECTED) + 1, 6, 70, 70
    th = int(T_ * 3 / 4)
    Wd, Hd = lw + ncol * (T_ + pad), head + len(G.ROWS) * (th + pad) + 40
    frames = []
    for i in range(STEPS_GRID * S + 1):
        step, frac = divmod(i / S, 1)
        step = int(step)
        img = np.full((Hd, Wd, 3), 255, np.uint8)
        text(img, "One latent action, applied to robot and human frames (sharp rendering)", (Wd // 2, 26), 0.7,
             thick=2, center=True)
        text(img, "real frame warped by the motion the latent action model predicts", (Wd // 2, 52), 0.5, MUTED,
             center=True)
        for ri, (role, _, _) in enumerate(G.ROWS):
            y = head + ri * (th + pad)
            text(img, role.capitalize(), (6, y + th // 2 + 6), 0.55)
            img[y:y + th, lw:lw + T_] = cv2.resize(cells[(ri, 1)][0], (T_, th))
            for ci in range(1, ncol):
                seq = cells[(ri, ci)]
                a, b = seq[min(step, STEPS_GRID)], seq[min(step + 1, STEPS_GRID)]
                x = lw + ci * (T_ + pad)
                img[y:y + th, x:x + T_] = cv2.resize((a * (1 - frac) + b * frac).astype(np.uint8), (T_, th))
        y = head + len(G.ROWS) * (th + pad) + 22
        text(img, "start frame", (lw + T_ // 2, y), 0.5, center=True)
        for ci, (lab, e, _) in enumerate(G.SELECTED, start=1):
            role = "robot" if src[e] == "robot_3cam" else "human"
            text(img, f"{lab} (from {role})", (lw + ci * (T_ + pad) + T_ // 2, y), 0.45, center=True)
        frames.append(img)
    clip = [frames[0]] * (FPS // 2) + frames + [frames[-1]] * FPS
    imageio.mimsave(FIG / "xemb_latent_grid_sharp.mp4", clip * 2, fps=FPS, quality=8, macro_block_size=8)


def trajectories(fr, pol):
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    scores, runs = [], []
    for name, er, tr, eh, th in PAIRS:
        for direction, (es, ts, et, tt) in (("human_to_robot", (eh, th, er, tr)), ("robot_to_human", (er, tr, eh, th))):
            mot = motion_seq(pol, fr, es, ts)
            dec, shp = drive(pol, fr, et, tt, mot)
            stat, _ = drive(pol, fr, et, tt, [torch.zeros_like(mot[0])] * K)
            src, tgt = real_seq(fr, es, ts), real_seq(fr, et, tt)
            src_full = [real_full(fr, es, ts + S * k) for k in range(K + 1)]
            tgt_full = [real_full(fr, et, tt + S * k) for k in range(K + 1)]
            for k in range(1, K + 1):
                vg = T.moving_flow(T.flow(stat[k], dec[k]))[0]
                vs = T.moving_flow(T.flow(src[0], src[k]))[0]
                vt = T.moving_flow(T.flow(tgt[0], tgt[k]))[0]
                scores.append(dict(pair=name, direction=direction, step=k, time_s=round(k * S / 30, 2),
                                   cos_generated_vs_source_real=round(T.cos(vg, vs), 4),
                                   cos_generated_vs_target_real_future=round(T.cos(vg, vt), 4),
                                   generated_motion_px=round(float(np.linalg.norm(vg)), 2)))
            for k in range(K + 1):
                for tag, im in (("source", src_full[k]), ("decoded", dec[k]), ("sharp", shp[k]), ("target_real", tgt_full[k])):
                    cv2.imwrite(str(TRAJ_DIR / f"{name}_{direction}_{tag}_k{k}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            runs.append((name, direction, src_full, dec, shp, tgt_full))
    s = pd.DataFrame(scores)
    s.to_csv(RES / "xemb_trajectory_transfer.csv", index=False)
    print(s.groupby(["pair", "direction"])[["cos_generated_vs_source_real", "cos_generated_vs_target_real_future"]]
          .mean().round(2).to_string())
    return runs, s


def traj_frames(run, size=300):
    name, direction, src, dec, shp, tgt = run
    a, b = direction.split("_to_")
    th = int(size * 3 / 4)
    pad, head, foot = 12, 92, 44
    Wd, Hd = 4 * size + 5 * pad, head + th + foot
    frames = []
    for i in range(K * S + 1):
        step, frac = divmod(i / S, 1)
        step = int(step)
        mix = lambda seq: (seq[min(step, K)] * (1 - frac) + seq[min(step + 1, K)] * frac).astype(np.uint8)
        img = np.full((Hd, Wd, 3), 255, np.uint8)
        text(img, f"The {a}'s latent sequence drives a {b} frame ({name.replace('_', ' ')})", (Wd // 2, 34), 0.75,
             thick=2, center=True)
        text(img, "one motion latent every 0.3 s, taken from the real demo on the left; 0.5x", (Wd // 2, 64), 0.5, MUTED,
             center=True)
        panels = [(f"real {a} demo (latent source)", src), (f"{b} + latents: model output", dec),
                  (f"{b} + latents: sharp rendering", shp), (f"real {b} demo (reference)", tgt)]
        for k, (lab, seq) in enumerate(panels):
            x = pad + k * (size + pad)
            img[head:head + th, x:x + size] = cv2.resize(mix([cv2.resize(q, (size, th)) for q in seq]), (size, th))
            text(img, lab, (x + size // 2, head + th + 28), 0.45, center=True)
        frames.append(img)
    return [frames[0]] * (FPS // 2) + frames + [frames[-1]] * FPS


def main():
    import types
    stage = "/home/admin_07/.claude/jobs/7ac23333/tmp/dk1"
    fr = T.Frames(str(Path.home() / "rlfv_stage/dk1-postprocessed-full-3cam-placeholder-20260719"))
    pol = T.load_policy("/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam2_43524549_030000",
                        "/home/admin_07/project_repos/lerobot_policy_lam_plain_dino/src")
    src, _ = G.episode_meta(types.SimpleNamespace(stage=stage))
    with torch.inference_mode():
        sharp_grid(fr, pol, src)
        runs, scores = trajectories(fr, pol)
    m = scores.groupby(["pair", "direction"]).cos_generated_vs_source_real.mean()
    allf = []
    for run in runs:
        allf += traj_frames(run)
    imageio.mimsave(FIG / "xemb_trajectory_transfer.mp4", allf, fps=FPS, quality=8, macro_block_size=8)
    best = m.sort_values(ascending=False).index[:3]
    for name, direction in best:
        run = next(r for r in runs if r[0] == name and r[1] == direction)
        imageio.mimsave(FIG / f"xemb_headline_trajectory_{name}_{direction}.mp4", traj_frames(run) * 3, fps=FPS,
                        quality=8, macro_block_size=8)
        print("headline", name, direction, round(m[(name, direction)], 3))
    print("mean cos generated vs source:", round(scores.cos_generated_vs_source_real.mean(), 3),
          "vs target real future:", round(scores.cos_generated_vs_target_real_future.mean(), 3))


if __name__ == "__main__":
    main()
