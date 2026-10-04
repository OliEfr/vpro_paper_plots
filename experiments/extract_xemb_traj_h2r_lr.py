#!/usr/bin/env python
"""Human -> robot trajectory transfer, LEFT/RIGHT movements, original pick-and-place eval tasks only.

Same method as extract_xemb_traj_h2r.py (a human demo's 4 motion latents, one per 0.3 s, drive a robot start
frame through the LAM decoder for 1.2 s; the robot clip really made the same movement and is the reference;
robot rows are raw decoder output), restricted to
  * the four original pick-and-place eval tasks: banana in cardboard box, banana in black bowl,
    milk on pink plate, salt on pink plate (human and robot clip both from these);
  * horizontal movements: the real motion of both clips, and the generated robot motion, within 30 deg of
    left-to-right or right-to-left in the front camera.
Pairs come from extract_xemb_evaltasks.py pairs_search (nearest raw-latent window, flow-verified).

Modes:  search  -> results/xemb_traj_h2r_lr_candidates.csv
        gallery -> figures/xemb_traj_h2r_lr_gallery.jpg
        dump    -> results/xemb_traj_h2r_lr.csv (+ per-row motion arrows), results/frames_xemb_traj_h2r_lr/
        video   -> figures/xemb_traj_h2r_lr.mp4
"""
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_xemb_realworld_transfer as T   # noqa: E402
import extract_xemb_traj_h2r as TH            # noqa: E402
import xemb_evaltasks_common as C             # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES, FIG = HERE / "results", HERE / "figures"
FRAME_DIR = RES / "frames_xemb_traj_h2r_lr"
K, S = TH.K, TH.S
HORIZ = 30.0     # deg

# (human gid, human frame, robot gid, robot frame) -- picked by eye from the gallery
EXAMPLES = [   # picked by eye from the gallery (search2 pool), distinct robot clips
    (1626, 100, 597, 130),   # milk on pink plate, right
    (803, 110, 503, 140),    # banana in cardboard box, right
    (847, 110, 551, 100),    # banana in black bowl, right
    (828, 120, 505, 160),    # banana in cardboard box, right
    (1636, 70, 609, 160),    # milk on pink plate, right
    (836, 120, 545, 120),    # banana in black bowl, right
    (825, 130, 492, 130),    # banana in cardboard box, right
    (1661, 120, 605, 90),    # milk on pink plate, left
]


def horizontal(ang):
    a = abs(((ang + 180) % 360) - 180)      # 0 = right, 180 = left
    return a <= HORIZ or a >= 180 - HORIZ


def ang_of(v):
    return float(np.degrees(np.arctan2(-v[1], v[0])))


def mode_search():
    tab, fr, pol = TH.setup()
    c = pd.read_csv(RES / "xemb_realworld_transfer_evaltasks_candidates.csv")
    c = c[(c.kind == "nearest") & c.task_r.isin(C.CORE_TASKS) & c.task_h.isin(C.CORE_TASKS)]
    c = c[(c.cos > 0.85) & (c.mag_r > 7) & (c.mag_h > 7) & c.ang.map(horizontal)]
    c = c.sort_values("dist").drop_duplicates(["gr", "tr"]).drop_duplicates(["gh", "th"])
    print("horizontal core-task pairs:", len(c), flush=True)
    rows = []
    for r in c.itertuples():
        if r.th + S * K + 9 >= fr.length(r.gh) or r.tr + S * K >= fr.length(r.gr):
            continue
        dec, stat, src, tgt = TH.run(pol, fr, r.gh, r.th, r.gr, r.tr)
        cs, ct, mn, mg = TH.score(dec, stat, src, tgt)
        vg = T.moving_flow(T.flow(stat[K], dec[K]))[0]
        vs = T.moving_flow(T.flow(src[0], src[K]))[0]
        vt = T.moving_flow(T.flow(tgt[0], tgt[K]))[0]
        rows.append(dict(gh=r.gh, th=r.th, gr=r.gr, tr=r.tr, task_h=r.task_h, task_r=r.task_r,
                         cos_vs_human=cs, cos_vs_robot_real=ct, min_cos_human=mn, motion_px=mg,
                         ang_human=ang_of(vs), ang_generated=ang_of(vg), ang_robot_real=ang_of(vt),
                         same_task=r.task_h == r.task_r))
    o = pd.DataFrame(rows)
    o["all_horizontal"] = o.ang_human.map(horizontal) & o.ang_generated.map(horizontal) & o.ang_robot_real.map(horizontal)
    o["score"] = o.cos_vs_human.clip(0) * o.cos_vs_robot_real.clip(0) * np.minimum(o.motion_px / 12, 1) * o.all_horizontal
    o.round(4).to_csv(RES / "xemb_traj_h2r_lr_candidates.csv", index=False)
    print(len(o), "runs;", int(o.all_horizontal.sum()), "fully horizontal;", int((o.score > 0.6).sum()), "with score > 0.6")


def mode_search2(stride=10, per_human=3):
    """Bigger pool without retrieval: every core-task window with real horizontal motion over 1.2 s; each human
    window is paired with same-task robot windows that really move the same way (left / right)."""
    tab, fr, pol = TH.setup()
    eps = tab[tab.task.isin(C.CORE_TASKS)]
    win = []
    for gid, r in eps.iterrows():
        L = int(r.length)
        for t in range(10, L - S * K - 12, stride):
            f = torch.nn.functional.interpolate(fr.get(gid, [t, t + S * K]).float().cuda() / 255.0, size=(256, 256))
            v, c = T.moving_flow(T.flow(T.u8(f[0]), T.u8(f[1])))
            if np.linalg.norm(v) < 12 or not horizontal(ang_of(v)):
                continue
            if r.source_kind == "robot_3cam" and not 25 < c[0] < 215:
                continue
            win.append(dict(gid=gid, t=t, kind=r.source_kind, task=r.task, vx=v[0], vy=v[1], cx=c[0], cy=c[1]))
    w = pd.DataFrame(win)
    w.round(3).to_csv(RES / "xemb_traj_h2r_lr_windows.csv", index=False)
    print("horizontal windows:", w.groupby("kind").size().to_dict(), flush=True)
    hum, rob = w[w.kind == "video_2cam"], w[w.kind == "robot_3cam"]
    rows = []
    for h in hum.itertuples():
        cand = rob[(rob.task == h.task) & (np.sign(rob.vx) == np.sign(h.vx))]
        if not len(cand):
            continue
        cand = cand.assign(d=np.hypot(cand.cx - h.cx, cand.cy - h.cy)).sort_values("d").drop_duplicates("gid").head(per_human)
        for r in cand.itertuples():
            dec, stat, src, tgt = TH.run(pol, fr, h.gid, h.t, r.gid, r.t)
            cs, ct, mn, mg = TH.score(dec, stat, src, tgt)
            vg = T.moving_flow(T.flow(stat[K], dec[K]))[0]
            rows.append(dict(gh=h.gid, th=h.t, gr=r.gid, tr=r.t, task_h=h.task, task_r=r.task, cos_vs_human=cs,
                             cos_vs_robot_real=ct, min_cos_human=mn, motion_px=mg, ang_human=ang_of((h.vx, h.vy)),
                             ang_generated=ang_of(vg), ang_robot_real=ang_of((r.vx, r.vy)), same_task=True))
    o = pd.DataFrame(rows)
    o["all_horizontal"] = o.ang_generated.map(horizontal)
    o["score"] = o.cos_vs_human.clip(0) * o.cos_vs_robot_real.clip(0) * np.minimum(o.motion_px / 12, 1) * o.all_horizontal
    o.round(4).to_csv(RES / "xemb_traj_h2r_lr_candidates.csv", index=False)
    print(len(o), "runs;", int(o.all_horizontal.sum()), "generated motion horizontal;", int((o.score > 0.6).sum()), "score > 0.6")
    print(o.sort_values("score", ascending=False).head(20).round(2).to_string())


def mode_gallery(top=24):
    tab, fr, pol = TH.setup()
    o = pd.read_csv(RES / "xemb_traj_h2r_lr_candidates.csv")
    blocks = []
    for r in o.sort_values("score", ascending=False).head(top).itertuples():
        dec, stat, src, tgt = TH.run(pol, fr, r.gh, r.th, r.gr, r.tr)
        b = np.ascontiguousarray(np.concatenate([TH.strip(src), TH.strip(dec), TH.strip(tgt),
                                                 np.full((10, 160 * (K + 1), 3), 255, np.uint8)], 0))
        cv2.putText(b, f"{r.gh}@{r.th} -> {r.gr}@{r.tr} {r.task_h[:20]} / {r.task_r[:20]} {r.cos_vs_human:.2f}/{r.cos_vs_robot_real:.2f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 1)
        blocks.append(b)
    blocks += [np.full_like(blocks[0], 255)] * (len(blocks) % 2)
    grid = np.concatenate([np.concatenate([blocks[i], np.full((blocks[i].shape[0], 16, 3), 255, np.uint8), blocks[i + 1]], 1)
                           for i in range(0, len(blocks), 2)], 0)
    imageio.imwrite(FIG / "xemb_traj_h2r_lr_gallery.jpg", grid, quality=88)


def mode_dump():
    tab, fr, pol = TH.setup()
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (gh, th, gr, tr) in enumerate(EXAMPLES):
        dec, stat, src, tgt = TH.run(pol, fr, gh, th, gr, tr)
        cs, ct, _, _ = TH.score(dec, stat, src, tgt)
        arrows = {"human": T.moving_flow(T.flow(src[0], src[K])), "robot_model": T.moving_flow(T.flow(stat[K], dec[K])),
                  "robot_real": T.moving_flow(T.flow(tgt[0], tgt[K]))}
        for k in range(K + 1):
            for tag, im in (("human", src[k]), ("robot_model", dec[k]), ("robot_real", tgt[k])):
                cv2.imwrite(str(FRAME_DIR / f"ex{i}_{tag}_k{k}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        row = dict(example=i, task_human=tab.loc[gh, "task"], task_robot=tab.loc[gr, "task"], human_episode=gh,
                   human_frame=th, robot_episode=gr, robot_frame=tr, cos_vs_human=round(cs, 4), cos_vs_robot_real=round(ct, 4))
        for tag, (v, c) in arrows.items():
            row.update({f"{tag}_cx": c[0] / 256, f"{tag}_cy": c[1] / 256, f"{tag}_u": v[0] / 256, f"{tag}_v": v[1] / 256})
        row["direction"] = "right" if arrows["human"][0][0] > 0 else "left"
        rows.append(row)
    d = pd.DataFrame(rows).round(4)
    d.to_csv(RES / "xemb_traj_h2r_lr.csv", index=False)
    print(d[["example", "task_human", "task_robot", "direction", "cos_vs_human", "cos_vs_robot_real"]].to_string())


def mode_video():
    TH.EXAMPLES = [("eval", gh, th, gr, tr) for gh, th, gr, tr in EXAMPLES]
    orig = TH.FIG
    tmp = RES / "_tmp_video"
    tmp.mkdir(exist_ok=True)
    TH.FIG = tmp
    TH.mode_video()
    TH.FIG = orig
    (tmp / "xemb_traj_h2r.mp4").rename(FIG / "xemb_traj_h2r_lr.mp4")
    tmp.rmdir()
    print("wrote", FIG / "xemb_traj_h2r_lr.mp4")


if __name__ == "__main__":
    {"search": mode_search, "search2": mode_search2, "gallery": mode_gallery, "dump": mode_dump, "video": mode_video}[sys.argv[1]]()
