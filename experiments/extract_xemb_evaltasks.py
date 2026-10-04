#!/usr/bin/env python
"""All xemb human <-> robot transfer results again, using ONLY the six DK1 real-hardware eval tasks.

Same methods as extract_xemb_realworld_transfer.py / extract_xemb_latent_grid.py /
extract_xemb_additional.py / make_xemb_videos.py (functions imported from them, selections overridden in
memory only), but every clip, latent, start frame and baseline comes from the eval tasks (see
xemb_evaltasks_common.py). Two of the six tasks (push milk to the right; banana in black bowl with a new
background) were not in the LAM's training data. All outputs carry the suffix ``_evaltasks``; the earlier
results are not touched.

Modes, in order (selections between them were picked by eye from the search outputs):
  latents       eval-task latents (+5 and +9 frames) for every frame -> ~/rlfv_stage/evaltasks_latents.npz
  pairs_search  robot 1 s windows (gripper in view, EE moves > 4 cm) -> nearest human window (raw +5-frame
                latent, 30 frames, every 3rd), verified by front-camera optical flow; + random baseline
                -> results/xemb_realworld_transfer_evaltasks_candidates.csv
  pairs_dump    figure inputs for PAIRS -> results/xemb_realworld_transfer_evaltasks{,_arrows,_similarity,_stats}.csv,
                results/frames_xemb_realworld_evaltasks/
  grid_search   candidate single latents (motion part of moving eval-task moments, +9 frames) scored on
                15 robot + 15 human eval-task start frames -> results/xemb_latent_grid_evaltasks_candidates.csv
  grid          GRID_COLS x GRID_ROWS: decoded cells (2 steps), sharp warped cells (3 steps), stats on 30
                random start frames -> results/xemb_latent_grid_evaltasks{,_stats,_latent_cos}.csv,
                results/frames_xemb_latent_grid_evaltasks{,_sharp}/
  traj          trajectory transfer on PAIRS, both directions -> results/xemb_trajectory_transfer_evaltasks.csv,
                results/frames_xemb_trajectory_evaltasks/
  videos        figures/xemb_*_evaltasks.mp4 (+ headline clips)

GPU + LAM stack (st-07: ~/miniconda3/envs/mg-latent/bin/python).
"""
import sys
import types
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
SUF = "_evaltasks" if C.VARIANT == "evaltasks" else f"_{C.VARIANT}"
W, SUB, STRIDE = 30, 3, 5

# ---- selections (filled after the searches; gid = episode id, see xemb_evaltasks_common.py) ----
PAIRS = [         # (movement, robot gid, robot frame, human gid, human frame) -- picked by eye from pairs_search
    ("right", 10009, 60, 11012, 55),     # push milk to the right: both push the milk to the right
    ("up_right", 563, 105, 13030, 40),   # banana in black bowl (robot) / same task, new background (human): carry up-right
    ("left", 12008, 20, 13009, 15),      # banana in black bowl, new background: both reach left for the banana
    ("down", 10003, 45, 11006, 50),      # push milk to the right: both come down, toward the camera
]
GRID_COLS = [     # (label, gid, frame) -- 3 robot + 3 human sources, picked from grid_search
    ("right", 10009, 101),      # robot, push milk to the right (task unseen in LAM training)
    ("up-right", 11049, 62),    # human, push milk to the right (unseen)
    ("away", 1687, 20),         # human, salt on pink plate
    ("up-left", 495, 48),       # robot, banana in cardboard box
    ("left", 607, 40),          # robot, milk on pink plate
    ("down-left", 871, 166),    # human, banana in black bowl
]
GRID_ROWS = [     # (role, gid, frame) -- start frames with the gripper / hand in view
    ("robot", 501, 63), ("robot", 498, 153), ("human", 1673, 76), ("human", 854, 101),
]


TRAJ_PAIRS = [    # trajectory-transfer pairs, top of traj_search (falls back to PAIRS when empty)
    ("milk_on_plate", 583, 145, 1645, 100),             # same task (milk on pink plate)
    ("salt_robot_banana_human", 675, 75, 848, 170),     # robot salt on pink plate / human banana in black bowl
    ("banana_box_robot_push_milk_human", 508, 100, 11006, 45),
    ("banana_in_box", 487, 110, 807, 70),               # same task (banana in cardboard box)
]

if C.VARIANT == "newtasks_lam4":   # sharedlam4 on the two new tasks only
    PAIRS = []        # retrieval within the two new tasks is not above chance with sharedlam4 (see candidates csv)
    TRAJ_PAIRS = [    # the new-task pairs of the eval-task figure, for a like-for-like comparison with sharedlam2
        ("push_milk_right", 10009, 60, 11012, 55),
        ("reach_left", 12008, 20, 13009, 15),
        ("push_milk_down", 10003, 45, 11006, 50),
    ]
    GRID_COLS = [
        ("right", 10004, 78),       # robot, push milk to the right
        ("up-right", 11045, 80),    # human, push milk to the right
        ("away", 12003, 71),        # robot, banana in black bowl (new background)
        ("left", 11028, 50),        # human, push milk to the right
        ("down-left", 13041, 45),   # human, banana in black bowl (new background)
        ("down", 10008, 73),        # robot, push milk to the right
    ]
    GRID_ROWS = [("robot", 12009, 37), ("robot", 12001, 99), ("human", 11029, 74), ("human", 13037, 39)]


# ----------------------------------------------------------------------------- helpers
def setup():
    tab = C.episode_table()
    return tab, C.MultiFrames(tab)


def latent_index():
    m, Z1, Z2 = C.load_latents()
    key = pd.Series(np.arange(len(m)), index=pd.MultiIndex.from_arrays([m.gid.values, m.frame.values]))
    return m, Z1, Z2, key


def source_means(m, Z, tab):
    rob = tab.loc[m.gid.values, "source_kind"].values == "robot_3cam"
    v = m.valid.values
    return {"robot": Z[rob & v].mean(0), "human": Z[~rob & v].mean(0)}


def windows(m, tab, states):
    """Start rows of valid 30-frame windows, their gid/frame, robot flag, and robot EE displacement."""
    starts = []
    for gid, idx in m.groupby("gid").indices.items():
        idx = np.sort(idx)
        for s in range(0, len(idx) - W, STRIDE):
            if m.valid.values[idx[s:s + W]].all():
                starts.append(idx[s])
    starts = np.array(starts)
    g, f = m.gid.values[starts], m.frame.values[starts]
    rob = tab.loc[g, "source_kind"].values == "robot_3cam"
    disp = np.zeros(len(starts))
    for i in np.where(rob)[0]:
        st = states.get(g[i])
        if st is not None and f[i] + W - 1 < len(st):
            disp[i] = np.linalg.norm(st[f[i] + W - 1, :3] - st[f[i], :3])
    return starts, g, f, rob, disp


def wflow(fr, gid, t):
    return T.window_flow(fr, gid, t)


# ----------------------------------------------------------------------------- modes
def mode_latents():
    tab, fr = setup()
    C.compute_latents(tab, C.load_policy(), fr)


def mode_pairs_search():
    tab, fr = setup()
    m, Z1, _, key = latent_index()
    states = C.robot_states(tab)
    starts, g, f, rob, disp = windows(m, tab, states)
    Fw = torch.tensor(np.concatenate([Z1[starts + o] for o in range(0, W, SUB)], 1)).cuda()
    hi = np.where(~rob)[0]
    qi = np.where(rob & (disp > 0.04))[0]
    print("windows", len(starts), "robot queries", len(qi), "human", len(hi), flush=True)
    rows = []

    def seqcorr(gr, tr, gh, th):
        zr = Z1[[key[(gr, tr + i)] for i in range(W)]]
        zh = Z1[[key[(gh, th + i)] for i in range(W)]]
        return float(np.corrcoef(zr.ravel(), zh.ravel())[0, 1])

    for n, q in enumerate(qi):
        vr, cr = wflow(fr, g[q], f[q])
        if np.linalg.norm(vr) < 6 or not cr[0] < 205:
            continue
        dist, j = torch.cdist(Fw[q:q + 1], Fw[hi]).min(1)
        h = hi[int(j)]
        vh, ch = wflow(fr, g[h], f[h])
        rows.append(dict(kind="nearest", gr=int(g[q]), tr=int(f[q]), gh=int(g[h]), th=int(f[h]),
                         task_r=tab.loc[g[q], "task"], task_h=tab.loc[g[h], "task"], dist=float(dist),
                         cos=T.cos(vr, vh), mag_r=float(np.linalg.norm(vr)), mag_h=float(np.linalg.norm(vh)),
                         rx=float(cr[0]), corr=seqcorr(g[q], f[q], g[h], f[h]),
                         ang=float(np.degrees(np.arctan2(-(vr + vh)[1], (vr + vh)[0])))))
        if n % 500 == 0:
            print("query", n, len(rows), flush=True)
    rng = np.random.default_rng(0)
    for _ in range(300):
        q, h = rng.choice(qi), rng.choice(hi)
        vr, _ = wflow(fr, g[q], f[q])
        vh, _ = wflow(fr, g[h], f[h])
        rows.append(dict(kind="random", gr=int(g[q]), tr=int(f[q]), gh=int(g[h]), th=int(f[h]),
                         task_r=tab.loc[g[q], "task"], task_h=tab.loc[g[h], "task"], cos=T.cos(vr, vh),
                         mag_r=float(np.linalg.norm(vr)), mag_h=float(np.linalg.norm(vh))))
    o = pd.DataFrame(rows)
    o.round(4).to_csv(RES / f"xemb_realworld_transfer{SUF}_candidates.csv", index=False)
    o["agree"] = (o.cos > 0.9) & (o.mag_r > 6) & (o.mag_h > 6)
    print(o.groupby("kind")[["cos", "agree"]].mean().round(3).to_string())


def mode_pairs_dump():
    tab, fr = setup()
    m, Z1, _, key = latent_index()
    MU = source_means(m, Z1, tab)
    out_dir = RES / f"frames_xemb_realworld{SUF}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lat, arrows, seqs = [], [], {}
    for name, gr, tr, gh, th in PAIRS:
        for role, gid, t in (("robot", gr, tr), ("human", gh, th)):
            z = Z1[[key[(gid, t + i)] for i in range(W)]]
            zc = z - MU[role]
            seqs[f"{name}:{role}"] = zc
            for i in range(W):
                lat.append(dict(pair=name, role=role, episode=gid, frame=t + i, step=i, task=tab.loc[gid, "task"],
                                **{f"z{k}": round(float(z[i, k]), 4) for k in range(8)},
                                **{f"zc{k}": round(float(zc[i, k]), 4) for k in range(8)}))
            for j, fi in enumerate(range(t, t + W + 1, W // 2)):
                img = fr.get(gid, [fi])[0].permute(1, 2, 0).numpy()
                cv2.imwrite(str(out_dir / f"{name}_{role}_k{j}.jpg"),
                            cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            v, c = wflow(fr, gid, t)
            arrows.append(dict(panel=name, role=role, kind="window", cx=c[0] / 256, cy=c[1] / 256, u=v[0] / 256, v=v[1] / 256))
    pd.DataFrame(lat).to_csv(RES / f"xemb_realworld_transfer{SUF}.csv", index=False)
    pd.DataFrame(arrows).round(4).to_csv(RES / f"xemb_realworld_transfer{SUF}_arrows.csv", index=False)
    names = list(seqs)
    sim = np.array([[np.corrcoef(seqs[x].ravel(), seqs[y].ravel())[0, 1] for y in names] for x in names])
    pd.DataFrame(sim, index=names, columns=names).round(3).to_csv(RES / f"xemb_realworld_transfer{SUF}_similarity.csv")
    c = pd.read_csv(RES / f"xemb_realworld_transfer{SUF}_candidates.csv")
    c["agree"] = (c.cos > 0.9) & (c.mag_r > 6) & (c.mag_h > 6)
    stats = []
    for k, gk in c.groupby("kind"):
        stats += [dict(metric=f"search_{k}_mean_cos", value=gk.cos.mean()), dict(metric=f"search_{k}_frac_agree", value=gk.agree.mean())]
    same = [sim[i, j] for i in range(len(names)) for j in range(i + 1, len(names)) if names[i].split(":")[0] == names[j].split(":")[0]]
    diff = [sim[i, j] for i in range(len(names)) for j in range(i + 1, len(names)) if names[i].split(":")[0] != names[j].split(":")[0]]
    stats += [dict(metric="similarity_same_movement_mean", value=float(np.mean(same))),
              dict(metric="similarity_different_movement_mean", value=float(np.mean(diff)))]
    s = pd.DataFrame(stats)
    s["value"] = s.value.round(4)
    s.to_csv(RES / f"xemb_realworld_transfer{SUF}_stats.csv", index=False)
    print(s.to_string())
    print(pd.DataFrame(sim, index=names, columns=names).round(2).to_string())


def score_rows(tab, n=15, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for kind, role in (("robot_3cam", "robot"), ("video_2cam", "human")):
        for gid in rng.choice(tab.index[tab.source_kind == kind].values, n, replace=False):
            rows.append((role, int(gid), int(tab.loc[gid, "length"] * rng.uniform(0.2, 0.6))))
    return rows


def mode_grid_search(ncand=300):
    tab, fr = setup()
    pol = C.load_policy()
    rng = np.random.default_rng(0)
    cands, motions = [], []
    with torch.inference_mode():
        for kind in ("robot_3cam", "video_2cam"):
            n = 0
            while n < ncand:
                gid = int(rng.choice(tab.index[tab.source_kind == kind].values))
                t = int(rng.integers(5, fr.length(gid) - 12))
                f2 = torch.nn.functional.interpolate(fr.get(gid, [t, t + 9]).float().cuda() / 255.0, size=(256, 256))
                vA, _ = T.moving_flow(T.flow(T.u8(f2[0]), T.u8(f2[1])))
                if np.linalg.norm(vA) < 4:
                    continue
                motions.append(G.motion_and_static(pol, G.clip(fr, gid, t))[0])
                cands.append(dict(src_gid=gid, src_frame=t, src=kind, task=tab.loc[gid, "task"], real_fx=vA[0], real_fy=vA[1]))
                n += 1
        rows = score_rows(tab)
        V = G.consistency(fr, pol, motions, rows)
    rob = np.array([r[0] == "robot" for r in rows])
    U = G.unit(V)
    mr, mh = U[:, rob].mean(1), U[:, ~rob].mean(1)
    c = pd.DataFrame(cands)
    c["R_robot"], c["R_human"] = np.linalg.norm(mr, axis=1), np.linalg.norm(mh, axis=1)
    c["cos_robot_human"] = (G.unit(mr) * G.unit(mh)).sum(1)
    c["angle_deg"] = np.degrees(np.arctan2(-(mr + mh)[:, 1], (mr + mh)[:, 0]))
    c["mag_robot"] = np.linalg.norm(V[:, rob], axis=-1).mean(1)
    c["mag_human"] = np.linalg.norm(V[:, ~rob], axis=-1).mean(1)
    c["score"] = c.R_robot * c.R_human * c.cos_robot_human.clip(0)
    c.round(4).to_csv(RES / f"xemb_latent_grid{SUF}_candidates.csv", index=False)
    # start-frame quality for the rows: how clearly the top latents read on each scoring frame
    top = c.sort_values("score", ascending=False).index[:40]
    agree = (U[top] * G.unit(U[top].mean(1))[:, None]).sum(-1)
    mag = np.linalg.norm(V[top], axis=-1)
    fs = pd.DataFrame(dict(role=[r[0] for r in rows], gid=[r[1] for r in rows], frame=[r[2] for r in rows],
                           task=[tab.loc[r[1], "task"] for r in rows], mean_agree=agree.mean(0), min_mag=mag.min(0), mean_mag=mag.mean(0)))
    fs.round(3).to_csv(RES / f"xemb_latent_grid{SUF}_startframes.csv", index=False)
    print(c.sort_values("score", ascending=False).head(40).round(3).to_string())
    print(fs.sort_values("mean_mag", ascending=False).round(2).to_string())


def mode_grid():
    tab, fr = setup()
    pol = C.load_policy()
    dec_dir, sharp_dir = RES / f"frames_xemb_latent_grid{SUF}", RES / f"frames_xemb_latent_grid{SUF}_sharp"
    dec_dir.mkdir(parents=True, exist_ok=True)
    sharp_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        motions = [G.motion_and_static(pol, G.clip(fr, gid, t))[0] for _, gid, t in GRID_COLS]
        for ri, (role, gid, t) in enumerate(GRID_ROWS):
            v = G.clip(fr, gid, t)
            _, zs = G.motion_and_static(pol, v)
            x0 = v[:, :1, :, 0]
            cv2.imwrite(str(dec_dir / f"r{ri}_c0.png"), cv2.cvtColor(T.u8(x0[0, 0]), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sharp_dir / f"r{ri}_c0.png"), cv2.cvtColor(A.real_full(fr, gid, t), cv2.COLOR_RGB2BGR))
            for ci, mm in enumerate(motions, start=1):
                img = T.u8(G.roll(pol, x0, zs + mm, G.STEPS)[0, 0])
                cv2.imwrite(str(dec_dir / f"r{ri}_c{ci}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                _, shp = A.drive(pol, fr, gid, t, [mm] * A.STEPS_GRID)
                cv2.imwrite(str(sharp_dir / f"r{ri}_c{ci}.png"), cv2.cvtColor(shp[-1], cv2.COLOR_RGB2BGR))
        rows = score_rows(tab, seed=11)          # fresh start frames, not the ones used to pick
        V = G.consistency(fr, pol, motions, rows)
    meta = [dict(kind="row", index=i, role=r, episode=gid, frame=t, task=tab.loc[gid, "task"]) for i, (r, gid, t) in enumerate(GRID_ROWS)]
    meta += [dict(kind="col", index=i + 1, role="robot" if tab.loc[gid, "source_kind"] == "robot_3cam" else "human", episode=gid,
                  frame=t, task=tab.loc[gid, "task"], label=lab) for i, (lab, gid, t) in enumerate(GRID_COLS)]
    pd.DataFrame(meta).to_csv(RES / f"xemb_latent_grid{SUF}.csv", index=False)
    rob = np.array([r[0] == "robot" for r in rows])
    U = G.unit(V)
    out = []
    for k, (lab, gid, t) in enumerate(GRID_COLS):
        mr, mh = U[k, rob].mean(0), U[k, ~rob].mean(0)
        ref = G.unit(mr + mh)
        out.append(dict(latent=lab, src_gid=gid, src_frame=t, src=tab.loc[gid, "source_kind"], consistency_robot_frames=np.linalg.norm(mr),
                        consistency_human_frames=np.linalg.norm(mh), cos_robot_vs_human=T.cos(mr, mh),
                        frac_frames_within_45deg=float(((U[k] @ ref) > np.cos(np.pi / 4)).mean()),
                        angle_robot_deg=np.degrees(np.arctan2(-mr[1], mr[0])), angle_human_deg=np.degrees(np.arctan2(-mh[1], mh[0]))))
    s = pd.DataFrame(out).round(3)
    s.to_csv(RES / f"xemb_latent_grid{SUF}_stats.csv", index=False)
    M = G.unit(torch.stack(motions).cpu().numpy())
    labs = [x[0] for x in GRID_COLS]
    pd.DataFrame(M @ M.T, index=labs, columns=labs).round(3).to_csv(RES / f"xemb_latent_grid{SUF}_latent_cos.csv")
    print(s.to_string())
    print(pd.DataFrame(M @ M.T, index=labs, columns=labs).round(2).to_string())


def mode_traj():
    tab, fr = setup()
    pol = C.load_policy()
    out_dir = RES / f"frames_xemb_trajectory{SUF}"
    out_dir.mkdir(parents=True, exist_ok=True)
    K, S = A.K, A.S
    scores = []
    with torch.inference_mode():
        for name, gr, tr, gh, th in (TRAJ_PAIRS or PAIRS):
            for direction, (gs, ts, gt, tt) in (("human_to_robot", (gh, th, gr, tr)), ("robot_to_human", (gr, tr, gh, th))):
                mot = A.motion_seq(pol, fr, gs, ts)
                dec, shp = A.drive(pol, fr, gt, tt, mot)
                stat, _ = A.drive(pol, fr, gt, tt, [torch.zeros_like(mot[0])] * K)
                src, tgt = A.real_seq(fr, gs, ts), A.real_seq(fr, gt, tt)
                for k in range(1, K + 1):
                    vg = T.moving_flow(T.flow(stat[k], dec[k]))[0]
                    vs = T.moving_flow(T.flow(src[0], src[k]))[0]
                    vt = T.moving_flow(T.flow(tgt[0], tgt[k]))[0]
                    scores.append(dict(pair=name, direction=direction, step=k, time_s=round(k * S / 30, 2),
                                       cos_generated_vs_source_real=round(T.cos(vg, vs), 4),
                                       cos_generated_vs_target_real_future=round(T.cos(vg, vt), 4),
                                       generated_motion_px=round(float(np.linalg.norm(vg)), 2)))
                for k in range(K + 1):
                    for tag, im in (("source", A.real_full(fr, gs, ts + S * k)), ("decoded", dec[k]), ("sharp", shp[k]),
                                    ("target_real", A.real_full(fr, gt, tt + S * k))):
                        cv2.imwrite(str(out_dir / f"{name}_{direction}_{tag}_k{k}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    s = pd.DataFrame(scores)
    s.to_csv(RES / f"xemb_trajectory_transfer{SUF}.csv", index=False)
    print(s.groupby(["pair", "direction"])[["cos_generated_vs_source_real", "cos_generated_vs_target_real_future"]].mean().round(2).to_string())
    print("mean:", s[["cos_generated_vs_source_real", "cos_generated_vs_target_real_future"]].mean().round(3).to_dict())


def mode_traj_search(n=40):
    """Trajectory transfer on the top verified pairs of pairs_search (both directions), ranked."""
    tab, fr = setup()
    pol = C.load_policy()
    c = pd.read_csv(RES / f"xemb_realworld_transfer{SUF}_candidates.csv")
    c = c[(c.kind == "nearest") & (c.cos > 0.9) & (c.mag_r > 8) & (c.mag_h > 8)]
    c = c.sort_values("corr", ascending=False).drop_duplicates(["gr"]).drop_duplicates(["gh"]).head(n)
    K, S = A.K, A.S
    rows = []
    with torch.inference_mode():
        for r in c.itertuples():
            for direction, (gs, ts, gt, tt) in (("human_to_robot", (r.gh, r.th, r.gr, r.tr)), ("robot_to_human", (r.gr, r.tr, r.gh, r.th))):
                if tt + S * K >= fr.length(gt) or ts + S * K + 9 >= fr.length(gs):
                    continue
                mot = A.motion_seq(pol, fr, gs, ts)
                dec, _ = A.drive(pol, fr, gt, tt, mot)
                stat, _ = A.drive(pol, fr, gt, tt, [torch.zeros_like(mot[0])] * K)
                src, tgt = A.real_seq(fr, gs, ts), A.real_seq(fr, gt, tt)
                cs, ct, mg = [], [], []
                for k in range(1, K + 1):
                    vg = T.moving_flow(T.flow(stat[k], dec[k]))[0]
                    cs.append(T.cos(vg, T.moving_flow(T.flow(src[0], src[k]))[0]))
                    ct.append(T.cos(vg, T.moving_flow(T.flow(tgt[0], tgt[k]))[0]))
                    mg.append(float(np.linalg.norm(vg)))
                rows.append(dict(gr=r.gr, tr=r.tr, gh=r.gh, th=r.th, task_r=r.task_r, task_h=r.task_h, direction=direction,
                                 ang=r.ang, cos_vs_source=np.mean(cs), cos_vs_target_real=np.mean(ct), min_cos_source=np.min(cs),
                                 mean_motion_px=np.mean(mg)))
    o = pd.DataFrame(rows)
    o["score"] = o.cos_vs_source.clip(0) * o.cos_vs_target_real.clip(0)
    o.round(4).to_csv(RES / f"xemb_trajectory_transfer{SUF}_candidates.csv", index=False)
    print(o.sort_values("score", ascending=False).head(25).round(2).to_string())


def pairs_video(fr, tab):
    m, Z1, _, key = latent_index()
    MU = source_means(m, Z1, tab)
    T.SELECTED = PAIRS
    lead, VW, VH, hm_h = 15, 400, 300, 64
    hm_w = 2 * VW
    allz = [Z1[[key[(g_, t + i)] for i in range(W)]] - MU[r] for _, gr, tr, gh, th in PAIRS for r, g_, t in (("robot", gr, tr), ("human", gh, th))]
    vmax = np.abs(np.concatenate(allz)).max()
    frames = []
    for name, gr, tr, gh, th in PAIRS:
        clips, zs = {}, {}
        for role, g_, t in (("robot", gr, tr), ("human", gh, th)):
            clips[role] = [cv2.resize(x.permute(1, 2, 0).numpy(), (VW, VH)) for x in fr.get(g_, list(range(t - lead, t + W + lead)))]
            zs[role] = Z1[[key[(g_, t + i)] for i in range(W)]] - MU[role]
        for i in range(len(clips["robot"])):
            img = np.full((60 + VH + 30 + 2 * hm_h + 50, 2 * VW + 30, 3), 255, np.uint8)
            MV.text(img, f"movement: {name.replace('_', '-')}   (eval tasks)", (img.shape[1] // 2, 30), 0.8, thick=2, center=True)
            MV.text(img, "0.5x", (img.shape[1] - 50, 30), 0.5, MV.MUTED)
            for k, role in enumerate(("robot", "human")):
                x = 10 + k * (VW + 10)
                img[60:60 + VH, x:x + VW] = clips[role][i]
                if lead <= i < lead + W:
                    cv2.rectangle(img, (x, 60), (x + VW - 1, 60 + VH - 1), (0, 114, 178), 4)
                g_ = gr if role == "robot" else gh
                MV.text(img, f"{role.capitalize()}: {tab.loc[g_, 'task']}", (x + 8, 84), 0.55, (255, 255, 255), 3)
                MV.text(img, f"{role.capitalize()}: {tab.loc[g_, 'task']}", (x + 8, 84), 0.55, MV.INK, 1)
            y0 = 60 + VH + 30
            MV.text(img, "latent (8 dims) over the 1 s window, minus each embodiment's average", (10, y0 - 8), 0.5, MV.MUTED)
            upto = int(np.clip(i - lead + 1, 0, W))
            for k, role in enumerate(("robot", "human")):
                hm = np.full((hm_h, hm_w, 3), 245, np.uint8)
                if upto:
                    col = cv2.resize(MV.rdbu(zs[role][:upto].T / vmax), (int(hm_w * upto / W), hm_h), interpolation=cv2.INTER_NEAREST)
                    hm[:, :col.shape[1]] = col
                yy = y0 + k * (hm_h + 4)
                img[yy:yy + hm_h, 30:30 + hm_w - 20] = hm[:, :hm_w - 20]
                MV.text(img, role[0].upper(), (10, yy + hm_h // 2 + 6), 0.55)
            frames.append(img)
        frames += [frames[-1]] * MV.FPS
    return frames


def mode_videos():
    import shutil
    import tempfile
    tab, fr = setup()
    pol = C.load_policy()
    src = tab.source_kind.to_dict()
    G.SELECTED, G.ROWS, T.SELECTED = GRID_COLS, GRID_ROWS, PAIRS
    kw = dict(fps=MV.FPS, quality=8, macro_block_size=8)
    heads = []
    tmp = Path(tempfile.mkdtemp())
    with torch.inference_mode():
        gv = MV.grid_video(fr, pol, src)
        for lab, gid, t in GRID_COLS[:3]:
            heads.append(MV.headline_latent(fr, pol, src, f"xemb_headline{SUF}_{lab.replace('-', '_')}", lab, 0, 2))
        # sharp grid video: A.sharp_grid writes fixed names, so point it at a temp dir and move the result
        A.GRID_DIR, A.FIG = tmp / "cells", tmp
        A.sharp_grid(fr, pol, src)
        shutil.move(str(tmp / "xemb_latent_grid_sharp.mp4"), str(FIG / f"xemb_latent_grid{SUF}_sharp.mp4"))
        runs = []
        for name, gr, tr, gh, th in (TRAJ_PAIRS or PAIRS):
            for direction, (gs, ts, gt, tt) in (("human_to_robot", (gh, th, gr, tr)), ("robot_to_human", (gr, tr, gh, th))):
                mot = A.motion_seq(pol, fr, gs, ts)
                dec, shp = A.drive(pol, fr, gt, tt, mot)
                srcf = [A.real_full(fr, gs, ts + A.S * k) for k in range(A.K + 1)]
                tgtf = [A.real_full(fr, gt, tt + A.S * k) for k in range(A.K + 1)]
                runs.append((name, direction, srcf, dec, shp, tgtf))
    imageio.mimsave(FIG / f"xemb_latent_grid{SUF}.mp4", gv, **kw)
    tv = []
    for r in runs:
        tv += A.traj_frames(r)
    imageio.mimsave(FIG / f"xemb_trajectory_transfer{SUF}.mp4", tv, **kw)
    sc = pd.read_csv(RES / f"xemb_trajectory_transfer{SUF}.csv").groupby(["pair", "direction"]).cos_generated_vs_source_real.mean()
    for name, direction in sc.sort_values(ascending=False).index[:3]:
        run = next(r for r in runs if r[0] == name and r[1] == direction)
        imageio.mimsave(FIG / f"xemb_headline_trajectory{SUF}_{name}_{direction}.mp4", A.traj_frames(run) * 3, **kw)
        heads.append(f"trajectory {name} {direction}")
    parts = [gv, tv]
    pv = None
    if PAIRS:
        pv = pairs_video(fr, tab)
        imageio.mimsave(FIG / f"xemb_realworld_transfer{SUF}.mp4", pv, **kw)
        parts.append(pv)
    size = (max(p[0].shape[1] for p in parts), max(p[0].shape[0] for p in parts))
    sub = ("real-world DK1 eval tasks only, multi-view latent action model" if C.VARIANT == "evaltasks" else
           "real-world DK1, the two new eval tasks, LAM sharedlam4 (trained on them)")
    allf = MV.title_card([("Same latent, same movement: robot and human", 0.9), (sub, 0.6)], size, MV.FPS * 3)
    if pv is not None:
        allf += MV.title_card([("1. Robot and human clips with the nearest latent", 0.8),
                               ("same movement -> same latent; different movement -> different latent", 0.55)], size, MV.FPS * 3)
        allf += MV.fit(pv, size)
    allf += MV.title_card([("One latent action, applied to robot and human frames", 0.8),
                           ("each column: one latent from one real demo moment; the model's decoder", 0.55),
                           ("moves the robot gripper and the human hand the same way", 0.55)], size, MV.FPS * 4)
    allf += MV.fit(gv, size)
    allf += MV.title_card([("A demo's latent sequence drives the other embodiment", 0.8),
                           ("one motion latent every 0.3 s; right: that embodiment's own real demo", 0.55)], size, MV.FPS * 3)
    allf += MV.fit(tv, size)
    imageio.mimsave(FIG / f"xemb_transfer_video{SUF}.mp4", allf, **kw)
    shutil.rmtree(tmp, ignore_errors=True)
    print("videos written with suffix", SUF, heads)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"latents": mode_latents, "pairs_search": mode_pairs_search, "pairs_dump": mode_pairs_dump,
     "grid_search": mode_grid_search, "grid": mode_grid, "traj": mode_traj, "traj_search": mode_traj_search,
     "videos": mode_videos}[mode]()
