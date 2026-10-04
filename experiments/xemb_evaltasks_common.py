"""Shared loaders for the EVAL-TASK-only versions of the xemb human <-> robot transfer results.

The six DK1 real-hardware eval tasks (verified 2026-10-03):
  core 4   -- in the canonical merge dk1-postprocessed-full-3cam-placeholder-20260719 (episode ids kept):
              banana in cardboard box, banana in black bowl (dark-green bowl), milk on pink plate
              (new-distractors variant), salt on pink plate
  extra 2  -- NOT in the LAM's training data, downloaded per repo from HF LearningFromVideo:
              push milk to the right; banana in black bowl with a NEW BACKGROUND (same label string as
              the core task, told apart by the source repo)
Episodes get one id space ("gid"): canonical episode_index for the core tasks, 10000 + 1000 * repo + episode
for the downloaded repos. Latents for every eval-task frame are computed here with the same LAM
(sharedlam2, ckpt 30k), both the +5-frame (idx1) and the +9-frame (idx2) latent, and cached.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_xemb_realworld_transfer as T   # noqa: E402

STAGE = "/home/admin_07/.claude/jobs/7ac23333/tmp/dk1"
CANON = str(Path.home() / "rlfv_stage/dk1-postprocessed-full-3cam-placeholder-20260719")
NEWROOT = Path.home() / "rlfv_stage/evaltasks_new"
import os
# VARIANT "evaltasks" (default): paper LAM sharedlam2 on all six eval tasks.
# VARIANT "newtasks_lam4": LAM sharedlam4 (44227723, trained on the new-tasks merge, which contains all six
# eval tasks incl. the two new ones) on the two NEW tasks only.
VARIANT = os.environ.get("XEMB_VARIANT", "evaltasks")
CACHE = Path.home() / f"rlfv_stage/{VARIANT}_latents.npz"
CKPT = {"evaltasks": "/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam2_43524549_030000",
        "newtasks_lam4": "/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam4_newtasks_44227723_030000"}[VARIANT]
LAM_SRC = "/home/admin_07/project_repos/lerobot_policy_lam_plain_dino/src"
CORE_TASKS = ["banana in cardboard box", "banana in black bowl", "milk on pink plate", "salt on pink plate"]
NEW_REPOS = [  # (repo dir, source kind, task label used here)
    ("dk1-push-milk-to-right-10eps-postprocessed", "robot_3cam", "push milk to the right"),
    ("dk1-push-milk-to-right-50eps-2cam-v2-postprocessed", "video_2cam", "push milk to the right"),
    ("dk1-banana-in-black-bowl-newbackground-10eps-postprocessed", "robot_3cam", "banana in black bowl (new background)"),
    ("dk1-banana-in-black-bowl-newbackground-50eps-2cam-v2-postprocessed", "video_2cam", "banana in black bowl (new background)"),
]
OFFS = [0, 1, 5, 9]


def episode_table():
    """One row per eval-task episode: gid, root, local episode, source kind, task, length, seen_in_lam_training."""
    s = pd.read_parquet(Path(STAGE) / "exports/sharedlam2/meta/source_episodes.parquet")
    s = s[s.task.isin(CORE_TASKS)]
    rows = [dict(gid=int(r.episode_index), root=CANON, ep=int(r.episode_index), source_kind=r.source_kind, task=r.task,
                 length=int(r.length), seen_in_lam_training=True) for r in s.itertuples()]
    for ri, (repo, kind, task) in enumerate(NEW_REPOS):
        root = str(NEWROOT / repo)
        e = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
        for r in e.itertuples():
            rows.append(dict(gid=10000 + 1000 * ri + int(r.episode_index), root=root, ep=int(r.episode_index),
                             source_kind=kind, task=task, length=int(r.length), seen_in_lam_training=False))
    t = pd.DataFrame(rows).set_index("gid")
    if VARIANT == "newtasks_lam4":
        t = t[~t.seen_in_lam_training]          # the two new tasks only
        t["seen_in_lam_training"] = True        # sharedlam4 did train on them
    return t


class MultiFrames:
    """Duck-types extract_xemb_realworld_transfer.Frames (get / length) over several dataset roots."""

    def __init__(self, table=None):
        self.table = episode_table() if table is None else table
        self.fr = {root: T.Frames(root) for root in self.table.root.unique()}

    def get(self, gid, idxs, cam="front"):
        r = self.table.loc[gid]
        return self.fr[r.root].get(int(r.ep), idxs, cam)

    def length(self, gid):
        return int(self.table.loc[gid, "length"])


def robot_states(table):
    """observation.state per (gid, frame) for robot episodes (EE x,y,z first)."""
    out = {}
    d, _, st = T.load_latents(STAGE)
    for gid in table.index[(table.source_kind == "robot_3cam") & table.seen_in_lam_training]:
        idx = np.where(d.episode_index.values == gid)[0]
        out[gid] = st[idx[np.argsort(d.frame_index.values[idx])]]
    for ri, (repo, kind, _) in enumerate(NEW_REPOS):
        if kind != "robot_3cam":
            continue
        p = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                       for f in sorted(glob.glob(str(NEWROOT / repo / "data/*/*.parquet")))])
        for ep, g in p.groupby("episode_index"):
            out[10000 + 1000 * ri + int(ep)] = np.stack(g.sort_values("frame_index")["observation.state"].values).astype(np.float32)
    return out


@torch.inference_mode()
def compute_latents(table, pol, fr, batch=48):
    """Per episode: decode front + side once, build every (t, t+1, t+5, t+9) clip, run the LAM encoder."""
    F = torch.nn.functional
    Z1, Z2, G_, Fi, V = [], [], [], [], []
    for n, gid in enumerate(table.index):
        L = fr.length(gid)
        views = []
        for cam in ("front", "side"):
            x = fr.get(gid, list(range(L)), cam)
            views.append(torch.cat([F.interpolate(x[i:i + 64].float().cuda() / 255.0, size=(256, 256), mode="bilinear",
                                                  align_corners=False) for i in range(0, L, 64)]))
        vv = torch.stack(views, 0)              # (V, L, 3, 256, 256)
        for t0 in range(0, L, batch):
            ts = list(range(t0, min(L, t0 + batch)))
            idx = torch.tensor([[min(t + o, L - 1) for o in OFFS] for t in ts], device="cuda")
            clip = vv[:, idx]                   # (V, B, 4, 3, H, W)
            clip = clip.permute(1, 0, 3, 2, 4, 5)   # (B, V, 3, 4, H, W)
            z = pol._extract_continuous_latents_from_video(clip).reshape(len(ts), 3, -1)
            Z1.append(z[:, 1].cpu().numpy()); Z2.append(z[:, 2].cpu().numpy())
            G_ += [gid] * len(ts); Fi += ts; V += [t + 9 <= L - 1 for t in ts]
        if n % 50 == 0:
            print("latents", n, "/", len(table), flush=True)
    np.savez(CACHE, gid=np.array(G_), frame=np.array(Fi), valid=np.array(V), z1=np.concatenate(Z1), z2=np.concatenate(Z2))


def load_latents():
    c = np.load(CACHE)
    return pd.DataFrame(dict(gid=c["gid"], frame=c["frame"], valid=c["valid"])), c["z1"].astype(np.float32), c["z2"].astype(np.float32)


def load_policy():
    return T.load_policy(CKPT, LAM_SRC)
