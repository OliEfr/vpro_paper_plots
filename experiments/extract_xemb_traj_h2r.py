#!/usr/bin/env python
"""Human -> robot trajectory transfer, many examples, training and eval tasks: dumps for plot_xemb_traj_h2r.py.

A HUMAN demo's latent sequence (4 motion latents, one per 0.3 s; motion part = clip latent minus the latent of
the same frame held still) drives a ROBOT start frame through the LAM decoder for 1.2 s. The robot clip is one
that really made the same movement (nearest latent window, verified by front-camera optical flow), so its own
real future is a reference the model never sees. The robot rows are raw decoder output.

Real-world DK1, multi-view LAM sharedlam2 (the paper's teacher). Splits:
  train -- non-eval tasks of the canonical merge (cached latent export)
  eval  -- the six eval tasks (latents from xemb_evaltasks_common.py; pairs from extract_xemb_evaltasks.py)

Modes:
  pairs    train-task pair search (robot window in view, EE moves > 4 cm -> nearest human window, raw +5-frame
           latent, flow-verified) -> results/xemb_traj_h2r_train_pairs.csv
  search   trajectory transfer (human -> robot) on the top verified pairs of both splits, scored against the
           human's real motion and the robot's real future -> results/xemb_traj_h2r_candidates.csv
  gallery  contact sheets of the best candidates -> figures/xemb_traj_h2r_gallery_{train,eval}.jpg
  dump     hand-picked EXAMPLES -> results/xemb_traj_h2r.csv, results/frames_xemb_traj_h2r/
  video    figures/xemb_traj_h2r.mp4

GPU + LAM stack (st-07: ~/miniconda3/envs/mg-latent/bin/python).
"""
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_xemb_additional as A           # noqa: E402
import extract_xemb_human2robot as H          # noqa: E402
import extract_xemb_realworld_transfer as T   # noqa: E402
import make_xemb_videos as MV                 # noqa: E402
import xemb_evaltasks_common as C             # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES, FIG = HERE / "results", HERE / "figures"
FRAME_DIR = RES / "frames_xemb_traj_h2r"
K, S, W = A.K, A.S, 30

# (split, human gid, human frame, robot gid, robot frame) -- hand-picked from the galleries
EXAMPLES = [   # picked by eye from the galleries (frames taken from the candidates csv)
    ("train", 1257, 165, 276, 195),     # push pink cup
    ("train", 1076, 140, 174, 130),     # can on stove
    ("train", 1313, 140, 304, 75),      # white cup on pan
    ("train", 998, 280, 114, 430),      # open drawer, blue brick in drawer, close drawer (closing)
    ("eval", 1645, 100, 583, 145),      # milk on pink plate
    ("eval", 807, 70, 487, 110),        # banana in cardboard box
    ("eval", 11009, 85, 10001, 125),    # push milk to the right
    ("eval", 1633, 105, 658, 80),       # milk on pink plate (human) / salt on pink plate (robot)
]


def setup():
    tab = H.full_table()
    return tab, C.MultiFrames(tab), C.load_policy()


def mode_pairs():
    """Train-task pairs: same procedure as extract_xemb_evaltasks.py pairs_search, on the cached canonical export."""
    tab, fr, _ = setup()
    d, Z, st = T.load_latents(C.STAGE)
    train_eps = set(tab.index[(tab.split == "train") & (tab.index < 10000)])
    keep = d.episode_index.isin(train_eps).values
    valid = d["latent_labels.valid"].values.astype(bool)
    rob = (d.source_kind == "robot_3cam").values
    starts = []
    for ep, idx in d[keep].groupby("episode_index").indices.items():
        idx = np.sort(np.where(keep)[0][idx])
        for s_ in range(0, len(idx) - W, 5):
            if valid[idx[s_:s_ + W]].all():
                starts.append(idx[s_])
    starts = np.array(starts)
    disp = np.linalg.norm(st[starts + W - 1, :3] - st[starts, :3], axis=1)
    Fw = torch.tensor(np.concatenate([Z[starts + o] for o in range(0, W, 3)], 1)).cuda()
    wr = rob[starts]
    qi, hi = np.where(wr & (disp > 0.04))[0], np.where(~wr)[0]
    print("train windows", len(starts), "robot queries", len(qi), flush=True)
    rows = []
    for n, q in enumerate(qi):
        gr, tr = int(d.episode_index[starts[q]]), int(d.frame_index[starts[q]])
        vr, cr = T.window_flow(fr, gr, tr)
        if np.linalg.norm(vr) < 8 or not 30 < cr[0] < 215:
            continue
        dist, j = torch.cdist(Fw[q:q + 1], Fw[hi]).min(1)
        h = starts[hi[int(j)]]
        gh, th = int(d.episode_index[h]), int(d.frame_index[h])
        vh, _ = T.window_flow(fr, gh, th)
        rows.append(dict(split="train", gr=gr, tr=tr, gh=gh, th=th, task_r=tab.loc[gr, "task"], task_h=tab.loc[gh, "task"],
                         dist=float(dist), cos=T.cos(vr, vh), mag_r=float(np.linalg.norm(vr)), mag_h=float(np.linalg.norm(vh))))
        if n % 500 == 0:
            print("query", n, len(rows), flush=True)
    o = pd.DataFrame(rows)
    o.round(4).to_csv(RES / "xemb_traj_h2r_train_pairs.csv", index=False)
    print("verified (cos > 0.9, both > 8 px):", ((o.cos > 0.9) & (o.mag_h > 8)).sum(), "of", len(o))


def candidate_pairs(n=70):
    tr = pd.read_csv(RES / "xemb_traj_h2r_train_pairs.csv")
    ev = pd.read_csv(RES / "xemb_realworld_transfer_evaltasks_candidates.csv")
    ev = ev[ev.kind == "nearest"].assign(split="eval")
    out = []
    for o in (tr, ev):
        o = o[(o.cos > 0.9) & (o.mag_r > 8) & (o.mag_h > 8)]
        o = o.sort_values("dist").drop_duplicates(["gr"]).drop_duplicates(["gh"]).head(n)
        out.append(o[["split", "gr", "tr", "gh", "th"]])
    return pd.concat(out, ignore_index=True)


@torch.inference_mode()
def run(pol, fr, gh, th, gr, tr):
    mot = A.motion_seq(pol, fr, gh, th)
    dec, _ = A.drive(pol, fr, gr, tr, mot)
    stat, _ = A.drive(pol, fr, gr, tr, [torch.zeros_like(mot[0])] * K)
    return dec, stat, A.real_seq(fr, gh, th), A.real_seq(fr, gr, tr)


def score(dec, stat, src, tgt):
    cs, ct, mg = [], [], []
    for k in range(1, K + 1):
        vg = T.moving_flow(T.flow(stat[k], dec[k]))[0]
        cs.append(T.cos(vg, T.moving_flow(T.flow(src[0], src[k]))[0]))
        ct.append(T.cos(vg, T.moving_flow(T.flow(tgt[0], tgt[k]))[0]))
        mg.append(float(np.linalg.norm(vg)))
    return np.mean(cs), np.mean(ct), np.min(cs), np.mean(mg)


def mode_search():
    tab, fr, pol = setup()
    rows = []
    for r in candidate_pairs().itertuples():
        if r.th + S * K + 9 >= fr.length(r.gh) or r.tr + S * K >= fr.length(r.gr):
            continue
        cs, ct, mn, mg = score(*run(pol, fr, r.gh, r.th, r.gr, r.tr))
        rows.append(dict(split=r.split, gh=r.gh, th=r.th, gr=r.gr, tr=r.tr, task_h=tab.loc[r.gh, "task"],
                         task_r=tab.loc[r.gr, "task"], cos_vs_human=cs, cos_vs_robot_real=ct, min_cos_human=mn,
                         motion_px=mg))
    o = pd.DataFrame(rows)
    o["score"] = o.cos_vs_human.clip(0) * o.cos_vs_robot_real.clip(0) * np.minimum(o.motion_px / 12, 1)
    o.round(4).to_csv(RES / "xemb_traj_h2r_candidates.csv", index=False)
    for split, g in o.groupby("split"):
        print(split, len(g), "runs; mean cos vs human", round(g.cos_vs_human.mean(), 3), "vs robot real",
              round(g.cos_vs_robot_real.mean(), 3), "| both > 0.8:", int(((g.cos_vs_human > 0.8) & (g.cos_vs_robot_real > 0.8)).sum()))


def strip(seq, size=160):
    return np.concatenate([cv2.resize(x, (size, size)) for x in seq], 1)


def mode_gallery(top=16):
    tab, fr, pol = setup()
    o = pd.read_csv(RES / "xemb_traj_h2r_candidates.csv")
    for split, g in o.groupby("split"):
        blocks = []
        for r in g.sort_values("score", ascending=False).head(top).itertuples():
            dec, stat, src, tgt = run(pol, fr, r.gh, r.th, r.gr, r.tr)
            b = np.ascontiguousarray(np.concatenate([strip(src), strip(dec), strip(tgt), np.full((10, 160 * (K + 1), 3), 255, np.uint8)], 0))
            cv2.putText(b, f"{r.gh}@{r.th} -> {r.gr}@{r.tr} {r.task_h[:22]} / {r.task_r[:22]} {r.cos_vs_human:.2f}/{r.cos_vs_robot_real:.2f}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)
            blocks.append(b)
        blocks += [np.full_like(blocks[0], 255)] * (len(blocks) % 2)
        grid = np.concatenate([np.concatenate([blocks[i], np.full((blocks[i].shape[0], 16, 3), 255, np.uint8), blocks[i + 1]], 1)
                               for i in range(0, len(blocks), 2)], 0)
        imageio.imwrite(FIG / f"xemb_traj_h2r_gallery_{split}.jpg", grid, quality=88)
        print("gallery", split)


def mode_dump():
    tab, fr, pol = setup()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (split, gh, th, gr, tr) in enumerate(EXAMPLES):
        dec, stat, src, tgt = run(pol, fr, gh, th, gr, tr)
        cs, ct, mn, mg = score(dec, stat, src, tgt)
        for k in range(K + 1):
            for tag, im in (("human", src[k]), ("robot_model", dec[k] if k else tgt[0]), ("robot_real", tgt[k])):
                cv2.imwrite(str(FRAME_DIR / f"ex{i}_{tag}_k{k}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        rows.append(dict(example=i, split=split, task_human=tab.loc[gh, "task"], task_robot=tab.loc[gr, "task"],
                         human_episode=gh, human_frame=th, robot_episode=gr, robot_frame=tr,
                         cos_vs_human=round(cs, 4), cos_vs_robot_real=round(ct, 4)))
    dd = pd.DataFrame(rows)
    dd.to_csv(RES / "xemb_traj_h2r.csv", index=False)
    print(dd.to_string())


def mode_video():
    tab, fr, pol = setup()
    B, pad, head, foot = 280, 12, 92, 44
    frames = []
    for split, gh, th, gr, tr in EXAMPLES:
        dec, _, _, _ = run(pol, fr, gh, th, gr, tr)
        hs = [cv2.resize(x.permute(1, 2, 0).numpy(), (B, B)) for x in fr.get(gh, list(range(th, th + S * K + 1)))]
        rs = [cv2.resize(x.permute(1, 2, 0).numpy(), (B, B)) for x in fr.get(gr, list(range(tr, tr + S * K + 1)))]
        dk = [rs[0]] + [cv2.resize(x, (B, B)) for x in dec[1:]]
        Wd, Hd = 3 * B + 4 * pad, head + B + foot
        clip = []
        for i in range(S * K + 1):
            step, frac = divmod(i / S, 1)
            step = int(step)
            m = (dk[min(step, K)] * (1 - frac) + dk[min(step + 1, K)] * frac).astype(np.uint8)
            img = np.full((Hd, Wd, 3), 255, np.uint8)
            MV.text(img, f"Human -> robot, {split} task: {tab.loc[gh, 'task']}", (Wd // 2, 34), 0.7, thick=2, center=True)
            MV.text(img, "one motion latent every 0.3 s from the human demo drives the robot frame; 0.5x", (Wd // 2, 64),
                    0.48, MV.MUTED, center=True)
            for k, (im, lab) in enumerate(((hs[i], "real human demo (latent source)"), (m, "robot + human latents (model)"),
                                           (rs[i], "real robot demo (reference)"))):
                x = pad + k * (B + pad)
                img[head:head + B, x:x + B] = im
                MV.text(img, lab, (x + B // 2, head + B + 28), 0.45, center=True)
            clip.append(img)
        frames += [clip[0]] * 8 + clip + [clip[-1]] * MV.FPS
    imageio.mimsave(FIG / "xemb_traj_h2r.mp4", frames, fps=MV.FPS, quality=8, macro_block_size=8)
    print("wrote", FIG / "xemb_traj_h2r.mp4")


if __name__ == "__main__":
    {"pairs": mode_pairs, "search": mode_search, "gallery": mode_gallery, "dump": mode_dump, "video": mode_video}[sys.argv[1]]()
