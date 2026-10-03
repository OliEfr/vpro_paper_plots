#!/usr/bin/env python
"""Write the dumps behind plot_xemb_realworld_transfer.py: human <-> robot latent transfer on DK1.

Question. Does the same LAM latent mean the same movement for a human hand and for the
robot? Real-world DK1 data, multi-view LAM sharedlam2 (43524549, ckpt 30k, front+side,
8-D continuous latent, idx1 = +5 frames), the paper's teacher.

Three modes, run in this order (the figure needs only ``dump``):

  search   1 s latent windows (30 frames, every 3rd latent -> 80-D). Every robot window
           whose end effector moves > 5 cm queries its nearest HUMAN window under plain
           L2 on the RAW latent (no normalisation). The top-N pairs, and N random
           robot/human pairs as a baseline, are checked against the video itself:
           Farneback optical flow on the front camera, first -> last frame of each window,
           mean over the top-3% magnitude pixels (= the moving hand / gripper). The two
           embodiments share the camera rig, so agreeing image motion = same movement.
           -> results/xemb_realworld_transfer_candidates.csv (+ a same-task pass)
  swap     Cross-embodiment latent swap through the LAM's own pixel decoder: a start frame
           of embodiment B is rolled out 6 x 5 frames (1 s) under the latent sequence of a
           clip of embodiment A, vs a static rollout under B's mean latent; the rendered
           motion is compared with A's true image motion. Controls: a random clip's latents.
           -> results/xemb_realworld_transfer_swap.csv
  dump     The figure inputs for the hand-picked pairs in SELECTED (four different movements):
           front key frames, per-clip flow arrows, the 30x8 latent sequences (raw `z*` and
           `zc*` = minus that embodiment's average latent, which removes the constant
           robot-vs-human offset), and the clip-by-clip similarity of the `zc` sequences
           (same movement -> similar, different movement -> different). Also the aggregate
           numbers (stats csv).

The pairs in SELECTED were picked by eye from the search output (cherry-picked by design:
the question is whether a clear example exists, the rates in the stats csv say how often).

Needs (NOT the vpro-plots env): torch, torchcodec, opencv, pandas, pyarrow, einops, the
LAM package (lerobot_policy_lam_plain_dino, md5-identical to the p19 training snapshot)
and a local GPU. On st-07: ~/miniconda3/envs/mg-latent/bin/python.

    python experiments/extract_xemb_realworld_transfer.py search --stage <stage>
    python experiments/extract_xemb_realworld_transfer.py swap   --stage <stage>
    python experiments/extract_xemb_realworld_transfer.py dump   --stage <stage>

<stage> holds exports/sharedlam2/ (the full-dataset latent export, see
fit_tsne_hardware_motion.py). --dataset-root is the canonical
dk1-postprocessed-full-3cam-placeholder-20260719 (front+side videos + meta), --ckpt the
sharedlam2 pretrained_model dir.
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
FRAME_DIR = RES / "frames_xemb_realworld"
FPS, W, SUB, STRIDE = 30, 30, 3, 5
OFFS = [0, 1, 5, 9]          # LAM frame offsets; latent idx1 = +5 frames

# (movement, robot episode, robot start frame, human episode, human start frame) -- picked by eye
# from the search: robot in view, real image motion of both clips agrees, four different movements.
SELECTED = [
    ("right", 272, 140, 1258, 195),      # push pink cup: both push the cup to the right
    ("up_right", 486, 115, 790, 95),     # banana in cardboard box: both lift the banana up-right over the box
    ("away", 112, 435, 997, 275),        # close drawer: both push the drawer shut, away from the camera
    ("down_right", 200, 140, 1645, 105), # robot (can on stove) and human (milk on plate) move down-right
]


# ----------------------------------------------------------------------------- data
def load_latents(stage):
    ex = Path(stage) / "exports" / "sharedlam2"
    fs = sorted(glob.glob(str(ex / "data" / "chunk-*" / "*.parquet")))
    cols = ["episode_index", "frame_index", "latent_labels.continuous_vector_latents",
            "latent_labels.valid", "observation.state"]
    d = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    src = pd.read_parquet(ex / "meta" / "source_episodes.parquet")[["episode_index", "source_kind", "task"]]
    d = d.merge(src, on="episode_index", how="left")
    Z = np.stack([v[0] for v in d["latent_labels.continuous_vector_latents"].values]).astype(np.float32)
    st = np.stack(d["observation.state"].values).astype(np.float32)
    d = d.drop(columns=["latent_labels.continuous_vector_latents", "observation.state"])
    return d, Z, st


class Frames:
    def __init__(self, root):
        self.root = root
        self.ep = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))]
                            ).set_index("episode_index")
        self._dec = {}

    def get(self, ep, idxs, cam="front"):
        from torchcodec.decoders import VideoDecoder
        r = self.ep.loc[ep]
        k = f"videos/observation.images.{cam}"
        path = f"{self.root}/videos/observation.images.{cam}/chunk-{int(r[k + '/chunk_index']):03d}/file-{int(r[k + '/file_index']):03d}.mp4"
        if path not in self._dec:
            if len(self._dec) > 8:
                self._dec.clear()
            self._dec[path] = VideoDecoder(path)
        t0 = float(r[k + "/from_timestamp"])
        return self._dec[path].get_frames_played_at(seconds=[t0 + i / FPS for i in idxs]).data  # N,3,H,W uint8

    def length(self, ep):
        return int(self.ep.loc[ep, "length"])


def to256(fr):
    return F.interpolate(fr.float().cuda() / 255.0, size=(256, 256), mode="bilinear", align_corners=False)


def u8(x):
    return (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def flow(a, b):
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(g(a), g(b), None, 0.5, 3, 15, 3, 5, 1.2, 0)


def moving_flow(f, q=0.97):
    """Mean flow and centroid over the top-(1-q) magnitude pixels (256x256 coords)."""
    mag = np.linalg.norm(f, axis=2)
    sel = mag >= max(np.quantile(mag, q), 0.5)
    if sel.sum() < 20:
        return np.zeros(2), np.array([np.nan, np.nan])
    ys, xs = np.nonzero(sel)
    return f[sel].mean(0), np.array([xs.mean(), ys.mean()])


def window_flow(fr, e, t):
    x = to256(fr.get(e, [t, t + W]))
    return moving_flow(flow(u8(x[0]), u8(x[1])))


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))


# ----------------------------------------------------------------------------- model
def load_policy(ckpt, lam_src):
    sys.path.insert(0, lam_src)
    import lerobot_policy_lam_plain as P
    assert P.__file__.startswith(lam_src), P.__file__
    from lerobot_policy_lam_plain.modeling_lam import LAMPolicy
    return LAMPolicy.from_pretrained(ckpt).to("cuda").eval()


@torch.inference_mode()
def decode(pol, first, z):
    """Pixel decoder of PlainLAMModel.forward, run on an arbitrary latent.
    first: (B,V,3,256,256) in [0,1]; z: (B,8) -> predicted +5-frame image (B,V,3,256,256)."""
    m = pol.lam
    B, V = first.shape[:2]
    act = m._prepare_action_tokens(m.bottleneck.decode(z.reshape(B, 1, -1).to(first.dtype)))
    bias = m.spatial_rel_pos_bias(m.grid_h, m.grid_w, device=first.device)
    fb = rearrange(first, "b v c h w -> (b v) c 1 h w")
    ctx = m.pixel_projection(fb)
    out = m.pixel_decoder(rearrange(ctx, "b t h w d -> (b t) (h w) d"), video_shape=tuple(ctx.shape[:-1]),
                          attn_bias=bias, context=rearrange(act, "b t h w d -> (b t) (h w) d").repeat_interleave(V, 0))
    out = rearrange(out, "(b t) (h w) d -> b t h w d", b=fb.shape[0], h=m.grid_h, w=m.grid_w)
    return rearrange(m.pixel_to_pixels(out), "(b v) c 1 h w -> b v c h w", b=B, v=V).clamp(0, 1)


def ridge_dirs(d, Z, st):
    """Robot-only ridge latent -> state[t+5]-state[t] (xyz); return the min-norm latent offset per +1 m."""
    rob = (d.source_kind == "robot_3cam").values
    valid = d["latent_labels.valid"].values.astype(bool)
    ep = d.episode_index.values
    nxt = np.clip(np.arange(len(d)) + 5, 0, len(d) - 1)
    m = rob & valid & (ep[nxt] == ep)
    X, Y = Z[m], st[nxt, :3][m] - st[m, :3]
    xm, ym = X.mean(0), Y.mean(0)
    Wt = np.linalg.solve((X - xm).T @ (X - xm) + np.eye(8), (X - xm).T @ (Y - ym))   # Ridge(alpha=1), (8,3)
    mu = np.stack([Z[rob & valid].mean(0), Z[~rob & valid].mean(0)])
    return np.linalg.pinv(Wt.T), mu   # (8,3), (2,8) [robot, human]


# ----------------------------------------------------------------------------- modes
def windows(d, st):
    valid = d["latent_labels.valid"].values.astype(bool)
    starts = []
    for _, idx in d.groupby("episode_index").indices.items():
        idx = np.sort(idx)
        for s in range(0, len(idx) - W, STRIDE):
            if valid[idx[s:s + W]].all():
                starts.append(idx[s])
    starts = np.array(starts)
    disp = np.linalg.norm(st[starts + W - 1, :3] - st[starts, :3], axis=1)
    return starts, disp


def mode_search(a):
    d, Z, st = load_latents(a.stage)
    fr = Frames(a.dataset_root)
    starts, disp = windows(d, st)
    rob = (d.source_kind == "robot_3cam").values[starts]
    task = d.task.values[starts]
    Fw = torch.tensor(np.concatenate([Z[starts + o] for o in range(0, W, SUB)], 1)).cuda()
    rows = []

    def verify(kind, r, h, dist):
        er, tr = int(d.episode_index[r]), int(d.frame_index[r])
        eh, th = int(d.episode_index[h]), int(d.frame_index[h])
        vr, _ = window_flow(fr, er, tr)
        vh, _ = window_flow(fr, eh, th)
        rows.append(dict(kind=kind, dist=dist, er=er, tr=tr, eh=eh, th=th, task_r=d.task[r], task_h=d.task[h],
                         disp_cm=round(100 * float(disp[np.searchsorted(starts, r)]), 2), cos=round(cos(vr, vh), 4),
                         mag_r=round(float(np.linalg.norm(vr)), 2), mag_h=round(float(np.linalg.norm(vh)), 2)))

    qi, hi = np.where(rob & (disp > 0.05))[0], np.where(~rob)[0]
    dv, dj = [], []
    for i in range(0, len(qi), 4096):
        v, j = torch.cdist(Fw[qi[i:i + 4096]], Fw[hi]).min(1)
        dv.append(v.cpu().numpy()); dj.append(j.cpu().numpy())
    dv, dj = np.concatenate(dv), np.concatenate(dj)
    seen = set()
    for o in np.argsort(dv):
        r, h = starts[qi[o]], starts[hi[dj[o]]]
        pair = (int(d.episode_index[r]), int(d.episode_index[h]))
        if pair in seen:
            continue
        seen.add(pair)
        verify("nearest", r, h, float(dv[o]))
        if len(seen) >= a.top:
            break
    for tk in np.unique(task):   # same-task pass: best 8 robot episodes per task
        q, hh = np.where(rob & (disp > 0.05) & (task == tk))[0], np.where(~rob & (task == tk))[0]
        if len(q) == 0 or len(hh) == 0:
            continue
        v, j = torch.cdist(Fw[q], Fw[hh]).min(1)
        v, j = v.cpu().numpy(), j.cpu().numpy()
        seen_r = set()
        for o in np.argsort(v):
            r = starts[q[o]]
            if int(d.episode_index[r]) in seen_r:
                continue
            seen_r.add(int(d.episode_index[r]))
            verify("same_task", r, starts[hh[j[o]]], float(v[o]))
            if len(seen_r) >= 8:
                break
    rng = np.random.default_rng(0)
    for _ in range(a.top):
        verify("random", starts[rng.choice(qi)], starts[rng.choice(hi)], float("nan"))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "xemb_realworld_transfer_candidates.csv", index=False)
    out["agree"] = (out.cos > 0.9) & (out.mag_r > 6) & (out.mag_h > 6)
    print(out.groupby("kind")[["cos", "agree"]].mean().round(3).to_string())


def mode_swap(a):
    d, Z, st = load_latents(a.stage)
    fr = Frames(a.dataset_root)
    pol = load_policy(a.ckpt, a.lam_src)
    _, mu = ridge_dirs(d, Z, st)
    MU = {"robot_3cam": mu[0], "video_2cam": mu[1]}
    valid = d["latent_labels.valid"].values.astype(bool)
    key = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_arrays([d.episode_index.values, d.frame_index.values]))
    src, task, L = (d.groupby("episode_index").source_kind.first(), d.groupby("episode_index").task.first(),
                    d.groupby("episode_index").size())
    K = 6

    def zseq(e, t, mapto=None):
        rows = [key.get((e, t + 5 * k)) for k in range(K)]
        if any(r is None for r in rows) or not valid[rows].all():
            return None
        z = Z[rows].copy()
        if mapto is not None:
            z = z - MU[src[e]] + MU[mapto]
        return torch.tensor(z).cuda()

    def last(x0, zs):
        x = x0
        for k in range(len(zs)):
            x = decode(pol, x, zs[k:k + 1])
        return u8(x[0, 0])

    rng = np.random.default_rng(0)
    cands = np.array([(e, t) for e in src.index for t in range(10, L[e] - 5 * K - 2, 10)])
    rng.shuffle(cands)
    res = []
    for eA, tA in cands:
        if len(res) >= a.nclip:
            break
        sA = src[eA]
        sB = "video_2cam" if sA == "robot_3cam" else "robot_3cam"
        fa = to256(fr.get(eA, [tA, tA + 5 * K]))
        vA, _ = moving_flow(flow(u8(fa[0]), u8(fa[1])))
        zc, zr = zseq(eA, tA, sB), zseq(eA, tA)
        if np.linalg.norm(vA) < 3 or zc is None:
            continue
        pool = [e for e in src.index if src[e] == sB and task[e] == task[eA]]
        for eB in rng.choice(pool, min(2, len(pool)), replace=False):
            tB = int(rng.integers(10, max(11, L[eB] - 10)))
            x0 = to256(fr.get(eB, [tB]))[0:1, None]
            S = last(x0, torch.tensor(np.tile(MU[sB], (K, 1))).cuda())
            while True:
                e2, t2 = cands[rng.integers(len(cands))]
                zctl = zseq(e2, t2, sB) if src[e2] == sA else None
                if zctl is not None:
                    break
            v_c, _ = moving_flow(flow(S, last(x0, zc)))
            v_r, _ = moving_flow(flow(S, last(x0, zr)))
            v_k, _ = moving_flow(flow(S, last(x0, zctl)))
            res.append(dict(eA=int(eA), tA=int(tA), srcA=sA, task=task[eA], eB=int(eB), tB=tB,
                            cos_raw=round(cos(v_r, vA), 4), cos_centred=round(cos(v_c, vA), 4),
                            cos_control=round(cos(v_k, vA), 4), mag_A=round(float(np.linalg.norm(vA)), 2)))
    out = pd.DataFrame(res)
    out.to_csv(RES / "xemb_realworld_transfer_swap.csv", index=False)
    print(out.groupby("srcA")[["cos_raw", "cos_centred", "cos_control"]].agg(["mean", "median"]).round(3).to_string())


def mode_dump(a):
    d, Z, st = load_latents(a.stage)
    fr = Frames(a.dataset_root)
    _, mu = ridge_dirs(d, Z, st)          # mu = per-embodiment average latent [robot, human]
    MU = {"robot": mu[0], "human": mu[1]}
    key = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_arrays([d.episode_index.values, d.frame_index.values]))
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAME_DIR.glob("*.jpg"):
        f.unlink()
    lat, arrows, seqs = [], [], {}
    for name, er, tr, eh, th in SELECTED:
        for role, e, t in (("robot", er, tr), ("human", eh, th)):
            z = Z[[key[(e, t + i)] for i in range(W)]]
            zc = z - MU[role]               # minus that embodiment's average latent
            seqs[f"{name}:{role}"] = zc
            for i in range(W):
                lat.append(dict(pair=name, role=role, episode=e, frame=t + i, step=i, task=d.task[key[(e, t)]],
                                **{f"z{k}": round(float(z[i, k]), 4) for k in range(8)},
                                **{f"zc{k}": round(float(zc[i, k]), 4) for k in range(8)}))
            for j, fi in enumerate(range(t, t + W + 1, W // 2)):   # start, middle, end
                img = fr.get(e, [fi])[0].permute(1, 2, 0).numpy()
                cv2.imwrite(str(FRAME_DIR / f"{name}_{role}_k{j}.jpg"),
                            cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            v, c = window_flow(fr, e, t)    # 256x256 coords -> stored as fractions of width/height
            arrows.append(dict(panel=name, role=role, kind="window", cx=c[0] / 256, cy=c[1] / 256,
                               u=v[0] / 256, v=v[1] / 256))
    pd.DataFrame(lat).to_csv(RES / "xemb_realworld_transfer.csv", index=False)
    pd.DataFrame(arrows).round(4).to_csv(RES / "xemb_realworld_transfer_arrows.csv", index=False)
    names = list(seqs)
    sim = np.array([[np.corrcoef(seqs[x].ravel(), seqs[y].ravel())[0, 1] for y in names] for x in names])
    pd.DataFrame(sim, index=names, columns=names).round(3).to_csv(RES / "xemb_realworld_transfer_similarity.csv")

    stats = []
    for fn, tag in ((RES / "xemb_realworld_transfer_candidates.csv", "search"),
                    (RES / "xemb_realworld_transfer_swap.csv", "swap")):
        if not fn.exists():
            continue
        c = pd.read_csv(fn)
        if tag == "search":
            c["agree"] = (c.cos > 0.9) & (c.mag_r > 6) & (c.mag_h > 6)
            for k, g in c.groupby("kind"):
                stats += [dict(metric=f"search_{k}_mean_cos", value=g.cos.mean()),
                          dict(metric=f"search_{k}_frac_agree", value=g.agree.mean())]
        else:
            for k, g in c.groupby("srcA"):
                for col in ("cos_raw", "cos_centred", "cos_control"):
                    stats.append(dict(metric=f"swap_from_{k}_{col}_mean", value=g[col].mean()))
    same = [sim[i, j] for i in range(len(names)) for j in range(len(names))
            if i < j and names[i].split(":")[0] == names[j].split(":")[0]]
    diff = [sim[i, j] for i in range(len(names)) for j in range(len(names))
            if i < j and names[i].split(":")[0] != names[j].split(":")[0]]
    stats += [dict(metric="similarity_same_movement_mean", value=float(np.mean(same))),
              dict(metric="similarity_different_movement_mean", value=float(np.mean(diff)))]
    s = pd.DataFrame(stats)
    s["value"] = s.value.round(4)
    s.to_csv(RES / "xemb_realworld_transfer_stats.csv", index=False)
    print(s.to_string())
    print(pd.DataFrame(sim, index=names, columns=names).round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["search", "swap", "dump"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dataset-root", default=str(Path.home() / "rlfv_stage/dk1-postprocessed-full-3cam-placeholder-20260719"))
    ap.add_argument("--ckpt", default="/home/admin_07/project_repos/rlfv_latent_eval/checkpoints/dk1_sharedlam2_43524549_030000")
    ap.add_argument("--lam-src", default="/home/admin_07/project_repos/lerobot_policy_lam_plain_dino/src")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--nclip", type=int, default=1500)
    a = ap.parse_args()
    {"search": mode_search, "swap": mode_swap, "dump": mode_dump}[a.mode](a)


if __name__ == "__main__":
    main()
