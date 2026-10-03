#!/usr/bin/env python
"""Write the dumps behind plot_xemb_latent_grid.py: ONE latent, applied to robot AND human frames.

LAPA-style latent-action grid (cf. latentactionpretraining.github.io, "latent_analysis_OpenX"),
on real-world DK1 with the paper's multi-view LAM (sharedlam2, 43524549, ckpt 30k).

A latent action here is the LAM latent of ONE real moment of a demo (the +9-frame latent,
0.3 s, front+side). It contains the movement but also some of the embodiment's look (robot
vs human latents are linearly separable), so only its MOTION PART is transferred:

    motion(A)      = z(clip of A at t)  -  z(static clip: A's frame t repeated 4x)
    applied to B   : decode(frame_B,  z_static(frame_B) + motion(A))

i.e. "what A did, relative to standing still", added to B's own "standing still". Two
decoder steps (feed the prediction back, same latent) = 0.6 s of motion.

Modes:
  search   1-step decodings of many candidate latents (moving moments of both embodiments)
           on 15 robot + 15 human start frames; per latent, how consistently its decoded
           motion (optical flow vs the static decoding) points the same way on robot frames,
           on human frames, and how well the two agree -> results/xemb_latent_grid_candidates.csv
  stats    the SELECTED latents on all 30 start frames: robot-vs-human direction agreement,
           fraction of frames within 45 deg, and the pairwise cosine of the motion latents
           -> results/xemb_latent_grid_stats.csv, results/xemb_latent_grid_latent_cos.csv
  render   the figure cells: real start frames (ROWS) and their decodings under each latent
           (SELECTED) -> results/frames_xemb_latent_grid/r{row}_c{col}.png (c0 = real frame)

SELECTED / ROWS were picked by eye among the top of the search (cherry-picked by design: one
clear latent per direction, start frames where the gripper / hand is in view).

Needs a GPU + the LAM stack (st-07: ~/miniconda3/envs/mg-latent/bin/python); shares its loaders
with extract_xemb_realworld_transfer.py.

    python experiments/extract_xemb_latent_grid.py {search,stats,render}
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_xemb_realworld_transfer import Frames, cos, decode, load_policy, moving_flow, flow, u8  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
CELL_DIR = RES / "frames_xemb_latent_grid"
OFFS = [0, 1, 5, 9]
IDX = 2            # latent for +9 frames
STEPS = 2          # decoder steps in the rendered grid

# (label, source episode, source frame) -- one real moment each; source embodiment from the dataset.
SELECTED = [("right", 276, 190), ("up-right", 1129, 96), ("away", 989, 345),
            ("up-left", 1602, 66), ("left", 425, 51), ("down-left", 538, 95)]
ROWS = [("robot", 403, 150), ("robot", 121, 32), ("human", 1151, 183), ("human", 1594, 212)]


def clip(fr, e, t, size=256):
    L = fr.length(e)
    idx = [min(t + o, L - 1) for o in OFFS]
    v = []
    for cam in ("front", "side"):
        x = torch.nn.functional.interpolate(fr.get(e, idx, cam).float().cuda() / 255.0, size=(size, size),
                                            mode="bilinear", align_corners=False)
        v.append(x.permute(1, 0, 2, 3))
    return torch.stack(v, 0)[None]          # (1,V,3,T,H,W)


@torch.inference_mode()
def lat(pol, v):
    return pol._extract_continuous_latents_from_video(v).reshape(v.shape[0], 3, -1)[:, IDX]


@torch.inference_mode()
def motion_and_static(pol, v):
    zs = lat(pol, v[:, :, :, :1].repeat(1, 1, 1, 4, 1, 1))[0]
    return lat(pol, v)[0] - zs, zs


@torch.inference_mode()
def roll(pol, x0, z, steps):
    x = x0
    for _ in range(steps):
        x = decode(pol, x, z[None] if z.ndim == 1 else z)
    return x


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)


def start_frames(fr, pol, rows):
    out = []
    for _, e, t in rows:
        v = clip(fr, e, t)
        _, zs = motion_and_static(pol, v)
        x0 = v[:, :1, :, 0]                  # front view, (1,1,3,H,W)
        out.append((x0, zs, u8(roll(pol, x0, zs, 1)[0, 0])))
    return out


def consistency(fr, pol, motions, rows):
    """V[k, i] = mean flow of latent k decoded (1 step) on start frame i, vs its static decoding."""
    S = start_frames(fr, pol, rows)
    M = torch.stack(motions)
    V = np.zeros((len(motions), len(rows), 2))
    for i, (x0, zs, ref) in enumerate(S):
        for k0 in range(0, len(M), 64):
            out = decode(pol, x0.repeat(len(M[k0:k0 + 64]), 1, 1, 1, 1), M[k0:k0 + 64] + zs)
            for j in range(out.shape[0]):
                V[k0 + j, i] = moving_flow(flow(ref, u8(out[j, 0])))[0]
    return V


def search_rows(a, fr, src):
    rng = np.random.default_rng(3)
    rows = []
    for s in ("robot_3cam", "video_2cam"):
        for e in rng.choice(src.index[src == s].values, 15, replace=False):
            rows.append(("robot" if s == "robot_3cam" else "human", int(e), int(fr.length(e) * rng.uniform(0.2, 0.6))))
    return rows


def episode_meta(a):
    ex = Path(a.stage) / "exports" / "sharedlam2" / "meta" / "source_episodes.parquet"
    m = pd.read_parquet(ex).set_index("episode_index")
    return m.source_kind, m.task


def mode_search(a):
    fr, pol = Frames(a.dataset_root), load_policy(a.ckpt, a.lam_src)
    src, task = episode_meta(a)
    rng = np.random.default_rng(0)
    cands, motions = [], []
    for s in ("robot_3cam", "video_2cam"):
        n = 0
        while n < a.ncand:
            e = int(rng.choice(src.index[src == s].values))
            t = int(rng.integers(5, fr.length(e) - 12))
            f2 = torch.nn.functional.interpolate(fr.get(e, [t, t + 9]).float().cuda() / 255.0, size=(256, 256))
            vA, _ = moving_flow(flow(u8(f2[0]), u8(f2[1])))
            if np.linalg.norm(vA) < 4:
                continue                       # skip idle moments
            m, _ = motion_and_static(pol, clip(fr, e, t))
            cands.append(dict(src_ep=e, src_frame=t, src=s, task=task[e], real_fx=vA[0], real_fy=vA[1]))
            motions.append(m)
            n += 1
    rows = search_rows(a, fr, src)
    V = consistency(fr, pol, motions, rows)
    rob = np.array([r[0] == "robot" for r in rows])
    U = unit(V)
    mr, mh = U[:, rob].mean(1), U[:, ~rob].mean(1)
    c = pd.DataFrame(cands)
    c["R_robot"], c["R_human"] = np.linalg.norm(mr, axis=1), np.linalg.norm(mh, axis=1)
    c["cos_robot_human"] = (unit(mr) * unit(mh)).sum(1)
    c["angle_deg"] = np.degrees(np.arctan2(-(mr + mh)[:, 1], (mr + mh)[:, 0]))   # 0 = right, 90 = up the image
    c["score"] = c.R_robot * c.R_human * c.cos_robot_human.clip(0)
    c.round(4).to_csv(RES / "xemb_latent_grid_candidates.csv", index=False)
    print(c.sort_values("score", ascending=False).head(30).round(3).to_string())


def mode_stats(a):
    fr, pol = Frames(a.dataset_root), load_policy(a.ckpt, a.lam_src)
    src, _ = episode_meta(a)
    motions = [motion_and_static(pol, clip(fr, e, t))[0] for _, e, t in SELECTED]
    rows = search_rows(a, fr, src)
    V = consistency(fr, pol, motions, rows)
    rob = np.array([r[0] == "robot" for r in rows])
    U = unit(V)
    out = []
    for k, (lab, e, t) in enumerate(SELECTED):
        mr, mh = U[k, rob].mean(0), U[k, ~rob].mean(0)
        ref = unit(mr + mh)
        out.append(dict(latent=lab, src_ep=e, src_frame=t, src=src[e], consistency_robot_frames=np.linalg.norm(mr),
                        consistency_human_frames=np.linalg.norm(mh), cos_robot_vs_human=cos(mr, mh),
                        frac_frames_within_45deg=float(((U[k] @ ref) > np.cos(np.pi / 4)).mean()),
                        angle_robot_deg=np.degrees(np.arctan2(-mr[1], mr[0])),
                        angle_human_deg=np.degrees(np.arctan2(-mh[1], mh[0]))))
    s = pd.DataFrame(out).round(3)
    s.to_csv(RES / "xemb_latent_grid_stats.csv", index=False)
    M = unit(torch.stack(motions).cpu().numpy())
    labs = [x[0] for x in SELECTED]
    pd.DataFrame(M @ M.T, index=labs, columns=labs).round(3).to_csv(RES / "xemb_latent_grid_latent_cos.csv")
    print(s.to_string())
    print(pd.DataFrame(M @ M.T, index=labs, columns=labs).round(2).to_string())


def mode_render(a):
    fr, pol = Frames(a.dataset_root), load_policy(a.ckpt, a.lam_src)
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    motions = [motion_and_static(pol, clip(fr, e, t))[0] for _, e, t in SELECTED]
    for ri, (role, e, t) in enumerate(ROWS):
        v = clip(fr, e, t)
        _, zs = motion_and_static(pol, v)
        x0 = v[:, :1, :, 0]
        cv2.imwrite(str(CELL_DIR / f"r{ri}_c0.png"), cv2.cvtColor(u8(x0[0, 0]), cv2.COLOR_RGB2BGR))
        for ci, m in enumerate(motions, start=1):
            img = u8(roll(pol, x0, zs + m, STEPS)[0, 0])
            cv2.imwrite(str(CELL_DIR / f"r{ri}_c{ci}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    src, task = episode_meta(a)
    rows = [dict(kind="row", index=i, role=r, episode=e, frame=t, task=task[e]) for i, (r, e, t) in enumerate(ROWS)]
    rows += [dict(kind="col", index=i + 1, role="robot" if src[e] == "robot_3cam" else "human", episode=e, frame=t,
                  task=task[e], label=lab) for i, (lab, e, t) in enumerate(SELECTED)]
    pd.DataFrame(rows).to_csv(RES / "xemb_latent_grid.csv", index=False)
    print(pd.DataFrame(rows).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["search", "stats", "render"])
    ap.add_argument("--stage", default="/home/admin_07/.claude/jobs/7ac23333/tmp/dk1")
    ap.add_argument("--dataset-root", default=str(Path.home() / "rlfv_stage/dk1-postprocessed-full-3cam-placeholder-20260719"))
    ap.add_argument("--ckpt", default="/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam2_43524549_030000")
    ap.add_argument("--lam-src", default="/home/admin_07/project_repos/lerobot_policy_lam_plain_dino/src")
    ap.add_argument("--ncand", type=int, default=300)
    a = ap.parse_args()
    {"search": mode_search, "stats": mode_stats, "render": mode_render}[a.mode](a)


if __name__ == "__main__":
    main()
