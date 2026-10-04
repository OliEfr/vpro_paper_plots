#!/usr/bin/env python
"""Human -> robot transfer, before / after: the dumps behind plot_xemb_human2robot.py.

For one HUMAN demo moment: its real "before" frame (t) and "after" frame (t + 0.6 s). The human's own
latents over that time (2 motion latents, one per 0.3 s = 9 frames; motion part = clip latent minus the
latent of the same frame held still) are applied by the LAM decoder to a ROBOT start frame of the same
task. The robot's "after" frame is the decoder's prediction (no post-processing). Does the robot move the
way the human moved?

Real-world DK1, multi-view LAM sharedlam2 (the paper's teacher). Two splits:
  train  -- tasks the LAM and the policies were trained on that are NOT eval tasks (canonical merge)
  eval   -- the six real-hardware eval tasks (core 4 in the canonical merge; push milk to the right and the
            new-background banana task from their HF repos, unseen by this LAM)

Modes:
  search   sample fast human moments per split, try several same-task robot start frames each, score by
           the flow direction / size of the robot's predicted motion (vs a "held still" rollout) against the
           human's real motion -> results/xemb_human2robot_candidates.csv
  gallery  contact sheets of the top candidates per split -> figures/xemb_human2robot_gallery_{train,eval}.jpg
  dump     the hand-picked EXAMPLES -> results/xemb_human2robot.csv, results/frames_xemb_human2robot/
  video    figures/xemb_human2robot.mp4

GPU + LAM stack (st-07: ~/miniconda3/envs/mg-latent/bin/python).
"""
import glob
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_xemb_additional as A           # noqa: E402
import extract_xemb_latent_grid as G          # noqa: E402
import extract_xemb_realworld_transfer as T   # noqa: E402
import make_xemb_videos as MV                 # noqa: E402
import xemb_evaltasks_common as C             # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES, FIG = HERE / "results", HERE / "figures"
FRAME_DIR = RES / "frames_xemb_human2robot"
K, S = 2, 9            # 2 motion latents x 9 frames = 0.6 s
EVAL_TASKS = set(C.CORE_TASKS) | {t for _, _, t in C.NEW_REPOS}

# (split, human gid, human frame, robot gid, robot frame) -- hand-picked from the gallery
EXAMPLES = [   # picked by eye from large-motion candidates (cos > 0.95, >= 18 px on both sides, robot in view)
    ("train", 1270, 136, 280, 157),     # push pink cup: both push right
    ("train", 1365, 270, 324, 250),     # blue brick in blue bowl (+ yellow brick): both lift up
    ("train", 683, 124, 17, 109),       # banana in basket: both reach down-left
    ("train", 1054, 39, 141, 81),       # can on grey plate: both move up-right
    ("eval", 11026, 74, 10004, 75),     # push milk to the right: both push right
    ("eval", 801, 77, 495, 99),         # banana in cardboard box: both lift up
    ("eval", 1621, 84, 601, 103),       # milk on pink plate: both move up-right
    ("eval", 1647, 105, 630, 144),      # milk on pink plate: both move down
]


def full_table():
    """All DK1 episodes: canonical merge (every task) + the two new eval-task repos."""
    s = pd.read_parquet(Path(C.STAGE) / "exports/sharedlam2/meta/source_episodes.parquet")
    rows = [dict(gid=int(r.episode_index), root=C.CANON, ep=int(r.episode_index), source_kind=r.source_kind, task=r.task,
                 length=int(r.length)) for r in s.itertuples()]
    for ri, (repo, kind, task) in enumerate(C.NEW_REPOS):
        root = str(C.NEWROOT / repo)
        e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
        rows += [dict(gid=10000 + 1000 * ri + int(r.episode_index), root=root, ep=int(r.episode_index), source_kind=kind,
                      task=task, length=int(r.length)) for r in e.itertuples()]
    t = pd.DataFrame(rows).set_index("gid")
    t["split"] = np.where(t.task.isin(EVAL_TASKS), "eval", "train")
    return t


def frames256(fr, gid, idxs):
    f = torch.nn.functional.interpolate(fr.get(gid, idxs).float().cuda() / 255.0, size=(256, 256), mode="bilinear")
    return [T.u8(x) for x in f]


@torch.inference_mode()
def transfer(pol, fr, hg, ht, rg, rt):
    mot = [G.motion_and_static(pol, G.clip(fr, hg, ht + S * k))[0] for k in range(K)]
    dec, _ = A.drive(pol, fr, rg, rt, mot)
    stat, _ = A.drive(pol, fr, rg, rt, [torch.zeros_like(mot[0])] * K)
    return dec, stat


def mode_search(n_human=180, n_robot=4):
    tab = full_table()
    fr = C.MultiFrames(tab)
    pol = C.load_policy()
    rng = np.random.default_rng(0)
    rows = []
    for split in ("train", "eval"):
        hum = tab[(tab.split == split) & (tab.source_kind == "video_2cam")]
        # always include the banana in cardboard box task among the eval humans
        n = 0
        tries = 0
        while n < n_human and tries < n_human * 20:
            tries += 1
            if split == "eval" and n < n_human // 6:
                pool = hum[hum.task == "banana in cardboard box"]
            else:
                pool = hum
            hg = int(rng.choice(pool.index.values))
            L = tab.loc[hg, "length"]
            if L < S * K + 20:
                continue
            ht = int(rng.integers(5, L - S * K - 10))
            h = frames256(fr, hg, [ht, ht + S * K])
            vh, ch = T.moving_flow(T.flow(h[0], h[1]))
            if np.linalg.norm(vh) < 12:            # fast, clearly visible human moments only
                continue
            robots = tab[(tab.task == tab.loc[hg, "task"]) & (tab.source_kind == "robot_3cam")]
            if not len(robots):
                continue
            n += 1
            for rg in rng.choice(robots.index.values, min(n_robot, len(robots)), replace=False):
                rL = tab.loc[rg, "length"]
                rt = int(np.clip(ht / L * rL + rng.integers(-15, 16), 5, rL - S * K - 10))
                dec, stat = transfer(pol, fr, hg, ht, int(rg), rt)
                vr, cr = T.moving_flow(T.flow(stat[K], dec[K]))
                rows.append(dict(split=split, task=tab.loc[hg, "task"], hg=hg, ht=ht, rg=int(rg), rt=rt,
                                 cos=T.cos(vh, vr), mag_h=float(np.linalg.norm(vh)), mag_r=float(np.linalg.norm(vr)),
                                 hx=ch[0], hy=ch[1], rx=cr[0], ry=cr[1],
                                 ang_h=float(np.degrees(np.arctan2(-vh[1], vh[0])))))
            if n % 30 == 0:
                print(split, n, flush=True)
    o = pd.DataFrame(rows)
    o["score"] = o.cos.clip(0) * np.minimum(o.mag_r / o.mag_h, o.mag_h / o.mag_r.clip(1e-3)).clip(0, 1) ** 0.5 * \
        np.minimum(o.mag_r / 10, 1)
    o.round(4).to_csv(RES / "xemb_human2robot_candidates.csv", index=False)
    for split, g in o.groupby("split"):
        print(split, "pairs", len(g), "mean cos", round(g.cos.mean(), 3), "frac cos>0.9", round((g.cos > 0.9).mean(), 3))


def arrow(img, v, c, gain=2.0, color=(230, 60, 20)):
    p0 = (int(c[0] - gain * v[0] / 2), int(c[1] - gain * v[1] / 2))
    p1 = (int(c[0] + gain * v[0] / 2), int(c[1] + gain * v[1] / 2))
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), 5, tipLength=0.3)
    cv2.arrowedLine(img, p0, p1, color, 2, tipLength=0.3)
    return img


def example_tiles(pol, fr, hg, ht, rg, rt):
    h = frames256(fr, hg, [ht, ht + S * K])
    r0 = frames256(fr, rg, [rt])[0]
    dec, stat = transfer(pol, fr, hg, ht, rg, rt)
    vh, ch = T.moving_flow(T.flow(h[0], h[1]))
    vr, cr = T.moving_flow(T.flow(stat[K], dec[K]))
    return h[0], h[1], r0, dec[K], (vh, ch), (vr, cr)


def mode_gallery(top=48):
    tab = full_table()
    fr = C.MultiFrames(tab)
    pol = C.load_policy()
    o = pd.read_csv(RES / "xemb_human2robot_candidates.csv")
    for split, g in o.groupby("split"):
        g = g.sort_values("score", ascending=False).drop_duplicates(["hg", "ht"]).head(top)
        rows = []
        for r in g.itertuples():
            h0, h1, r0, r1, (vh, ch), (vr, cr) = example_tiles(pol, fr, r.hg, r.ht, r.rg, r.rt)
            h1, r1 = arrow(h1.copy(), vh, ch), arrow(r1.copy(), vr, cr)
            tile = np.concatenate([h0, h1, np.full((256, 12, 3), 255, np.uint8), r0, r1], 1)
            tile = np.ascontiguousarray(tile)
            cv2.putText(tile, f"{r.hg}@{r.ht} -> {r.rg}@{r.rt}  {r.task[:28]}  cos {r.cos:.2f}", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            rows.append(tile)
        cols = 2
        rows += [np.full_like(rows[0], 255)] * ((-len(rows)) % cols)
        grid = np.concatenate([np.concatenate(rows[i:i + cols], 1) for i in range(0, len(rows), cols)], 0)
        imageio.imwrite(FIG / f"xemb_human2robot_gallery_{split}.jpg", grid[::2, ::2], quality=88)
        print("gallery", split, len(g))


def mode_dump():
    tab = full_table()
    fr = C.MultiFrames(tab)
    pol = C.load_policy()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (split, hg, ht, rg, rt) in enumerate(EXAMPLES):
        h0, h1, r0, r1, (vh, ch), (vr, cr) = example_tiles(pol, fr, hg, ht, rg, rt)
        for tag, im in (("human_before", h0), ("human_after", h1), ("robot_before", r0), ("robot_after_model", r1)):
            cv2.imwrite(str(FRAME_DIR / f"ex{i}_{tag}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        rows.append(dict(example=i, split=split, task_human=tab.loc[hg, "task"], task_robot=tab.loc[rg, "task"],
                         human_episode=hg, human_frame=ht, robot_episode=rg, robot_frame=rt, seconds=K * S / 30,
                         cos_direction=round(T.cos(vh, vr), 4),
                         human_cx=ch[0] / 256, human_cy=ch[1] / 256, human_u=vh[0] / 256, human_v=vh[1] / 256,
                         robot_cx=cr[0] / 256, robot_cy=cr[1] / 256, robot_u=vr[0] / 256, robot_v=vr[1] / 256))
    d = pd.DataFrame(rows)
    d.round(4).to_csv(RES / "xemb_human2robot.csv", index=False)
    print(d[["example", "split", "task_human", "task_robot", "cos_direction"]].to_string())


def mode_video():
    tab = full_table()
    fr = C.MultiFrames(tab)
    pol = C.load_policy()
    B = 300
    frames = []
    for split, hg, ht, rg, rt in EXAMPLES:
        hseq = [cv2.resize(x.permute(1, 2, 0).numpy(), (B, B)) for x in fr.get(hg, list(range(ht, ht + S * K + 1)))]
        with torch.inference_mode():
            mot = [G.motion_and_static(pol, G.clip(fr, hg, ht + S * k))[0] for k in range(K)]
            dec, _ = A.drive(pol, fr, rg, rt, mot)
        r0 = cv2.resize(frames256(fr, rg, [rt])[0], (B, B))
        dec = [r0] + [cv2.resize(x, (B, B)) for x in dec[1:]]
        pad, head, foot = 14, 92, 44
        Wd, Hd = 2 * B + 3 * pad + 40, head + B + foot
        clip = []
        for i in range(S * K + 1):
            step, frac = divmod(i / S, 1)
            step = int(step)
            a, b = dec[min(step, K)], dec[min(step + 1, K)]
            img = np.full((Hd, Wd, 3), 255, np.uint8)
            MV.text(img, f"Human -> robot ({split} task: {tab.loc[hg, 'task']})", (Wd // 2, 34), 0.7, thick=2, center=True)
            MV.text(img, "the human's latents, applied by the latent action model to a robot frame; 0.5x", (Wd // 2, 64),
                    0.48, MV.MUTED, center=True)
            img[head:head + B, pad:pad + B] = hseq[i]
            cv2.arrowedLine(img, (pad + B + 8, head + B // 2), (pad + B + 34, head + B // 2), MV.INK, 2, tipLength=0.4)
            x = 2 * pad + B + 40
            img[head:head + B, x:x + B] = (a * (1 - frac) + b * frac).astype(np.uint8)
            MV.text(img, "real human demo", (pad + B // 2, head + B + 28), 0.55, center=True)
            MV.text(img, "robot + human's latents (model output)", (x + B // 2, head + B + 28), 0.5, center=True)
            clip.append(img)
        frames += ([clip[0]] * 8 + clip + [clip[-1]] * MV.FPS) * 2
    imageio.mimsave(FIG / "xemb_human2robot.mp4", frames, fps=MV.FPS, quality=8, macro_block_size=8)
    print("wrote", FIG / "xemb_human2robot.mp4")


if __name__ == "__main__":
    {"search": mode_search, "gallery": mode_gallery, "dump": mode_dump, "video": mode_video}[sys.argv[1]]()
