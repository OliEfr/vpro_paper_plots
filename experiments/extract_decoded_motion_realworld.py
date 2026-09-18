#!/usr/bin/env python
"""Write results/decoded_motion_realworld.csv and results/frames_realworld/*.jpg.

Inputs (staged from MN5, Felix's account, read-only):

  <stage>/exports/sharedlam2/        full-dataset latent export of the multi-view LAM
                                     (dk1_postproc19_sharedlam2_43524549_ckpt030000_idx1)
  <stage>/exports/sharedlam3side/    same for the side-only LAM
                                     (dk1_postproc19_sharedlam3_44160743_ckpt030000_idx1)
  <stage>/dataset/meta/              meta of experiments/dk1-postprocessed-full-3cam-placeholder-20260719
  <stage>/videos/{front,side}/file-XXX.mp4   the video files holding the selected episodes

Both exports label the identical 411,445 frames (1,721 episodes: 681 robot_3cam with
7-D end-effector actions, 1,040 video_2cam human episodes without actions).

Method. Per LAM: StandardScaler + Ridge(alpha=1) and MLP(512,256) fitted on ALL valid
robot_3cam rows (latent -> 7-D current action), then applied to every valid row, so the
human episodes receive a decoded action from a probe that never saw human video. A
20 %-of-episodes holdout of the robot rows is scored first and printed; it reproduces the
cluster probe's ridge R^2 (0.437 multi-view, 0.416 side-only).

The CSV keeps only the SELECTED episodes (one robot + one human episode per task), with
columns episode_index, frame_index, task, source_kind, row_label, gt_{x,y,z,wx,wy,wz,grip}
(NaN for human rows) and {ours_multi,ours_single}_{ridge,mlp}_{x,...}; the same with a `d`
prefix (gt_dx, ours_multi_ridge_dx, ...) for the 5-frame EE-motion target state[t+5]-state[t]. Key frames are cut
with ffmpeg at 0/25/50/75/97 % of the episode from the front camera (320x240).

Run:
    python experiments/extract_decoded_motion_realworld.py --stage <stage> \
        --decoded <stage>/decoded_sharedlam2.parquet <stage>/decoded_sharedlam3side.parquet
where the decoded parquets are the output of the fit step (fit_decode.py in the stage dir,
kept in this folder as fit_decode_realworld.py).
"""
import argparse
import glob
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT_CSV = HERE / "results" / "decoded_motion_realworld.csv"
OUT_FRAMES = HERE / "results" / "frames_realworld"

# (task, source_kind, row label). One robot reference row per task, then the human row.
SELECT = [
    ("push pink cup", "robot_3cam", "robot demo\npush cup right"),
    ("push pink cup", "video_2cam", "human video\npush cup right"),
    ("milk on pink plate", "robot_3cam", "robot demo\nmilk on plate"),
    ("milk on pink plate", "video_2cam", "human video\nmilk on plate"),
]
# --gallery: 2 robot + 2 human episodes per task. Only the four real-world EVAL tasks
# (novel object: milk / salt on pink plate; novel placement: banana in cardboard box / black
# bowl), whose 50+50 episodes were held out of LAM training (dk1_heldout_evalprobe_20260826).
GALLERY_TASKS = ["milk on pink plate", "salt on pink plate", "banana in cardboard box", "banana in black bowl"]
DIMS = ["x", "y", "z", "wx", "wy", "wz", "grip"]
FPS = 30.0
KEYS = (0.0, 0.25, 0.5, 0.75, 0.97)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--multi", type=Path, help="decoded parquet, multi-view LAM")
    ap.add_argument("--single", type=Path, help="decoded parquet, side-only LAM")
    ap.add_argument("--episode-offset", type=int, default=0,
                    help="pick the n-th episode of each (task, source) instead of the first")
    ap.add_argument("--gallery", action="store_true",
                    help="instead of SELECT: for each task in GALLERY_TASKS the first two ROBOT and the "
                         "first two HUMAN episodes (robot, robot, human, human); writes *_gallery.csv "
                         "so examples can be picked side by side")
    ap.add_argument("--selected", type=Path, default=None,
                    help="results/decoded_motion_realworld_selection.csv from select_decoded_motion_realworld.py: "
                         "for each eval task the --top-k robot and human episodes by the reliability proxy "
                         "(robot rows first); writes *_selected.csv")
    ap.add_argument("--top-k", type=int, default=2)
    a = ap.parse_args()
    multi = pd.read_parquet(a.multi or a.stage / "decoded_sharedlam2.parquet")
    single = pd.read_parquet(a.single or a.stage / "decoded_sharedlam3side.parquet")

    ep_meta = pd.concat([pd.read_parquet(f) for f in sorted(
        glob.glob(str(a.stage / "dataset/meta/episodes/**/*.parquet"), recursive=True))])
    src = pd.read_parquet(a.stage / "exports/sharedlam2/meta/source_episodes.parquet")
    meta = ep_meta.merge(src[["episode_index", "task", "source_kind"]], on="episode_index")

    OUT_FRAMES.mkdir(parents=True, exist_ok=True)
    select = SELECT
    out_csv = OUT_CSV
    if a.gallery:
        import textwrap
        select = []
        for t in GALLERY_TASKS:
            short = "\n".join(textwrap.wrap(t, 16))
            for kind, klabel in (("robot_3cam", "robot"), ("video_2cam", "human")):
                for k in range(2):
                    select.append((t, kind, f"{klabel} {k + 1}|{short}", k))
        out_csv = OUT_CSV.with_name(OUT_CSV.stem + "_gallery.csv")
    elif a.selected:
        import textwrap
        sel = pd.read_csv(a.selected)
        sel = sel[sel.rank_in_task <= a.top_k]
        select = []
        # task order = the order in the selection CSV's task column (eval or train-best set)
        for t in sel.task.drop_duplicates().tolist() if not set(sel.task) <= set(GALLERY_TASKS) else GALLERY_TASKS:
            short = "\n".join(textwrap.wrap(t, 16))
            for kind, klabel in (("robot_3cam", "robot"), ("video_2cam", "human")):
                g = sel[(sel.task == t) & (sel.source_kind == kind)].sort_values("rank_in_task")
                for _, r in g.iterrows():
                    # negative offset = explicit episode index (resolved below)
                    select.append((t, kind, f"{klabel} {int(r.rank_in_task)}|{short}", -int(r.episode_index) - 1))
        out_csv = OUT_CSV.with_name(OUT_CSV.stem + ("_selected_train.csv" if "train" in a.selected.stem else "_selected.csv"))
    else:
        select = [(t, k, l, a.episode_offset) for t, k, l in SELECT]
    rows = []
    for task, kind, label, offset in select:
        cand = meta[(meta.task == task) & (meta.source_kind == kind)].sort_values("episode_index")
        m = cand.iloc[offset] if offset >= 0 else cand[cand.episode_index == -offset - 1].iloc[0]
        ep = int(m.episode_index)
        mm = multi[multi.episode_index == ep].sort_values("frame_index")
        ss = single[single.episode_index == ep].sort_values("frame_index")
        assert len(mm) == len(ss) == int(m.length), (ep, len(mm), len(ss), m.length)
        d = pd.DataFrame({"episode_index": ep, "frame_index": mm.frame_index.to_numpy(),
                          "task": task, "source_kind": kind, "row_label": label,
                          "valid": mm.valid.to_numpy()})
        # two targets: "" = absolute EE pose target (the stored action), "d" = realized
        # EE motion state[t+5]-state[t] over the LAM horizon (see fit_decode_realworld.py)
        for pre in ("", "d"):
            for k in DIMS:
                d[f"gt_{pre}{k}"] = mm[f"gt_{pre}{k}"].to_numpy() if kind == "robot_3cam" else np.nan
                for probe in ("ridge", "mlp"):
                    d[f"ours_multi_{probe}_{pre}{k}"] = mm[f"{probe}_{pre}{k}"].to_numpy()
                    d[f"ours_single_{probe}_{pre}{k}"] = ss[f"{probe}_{pre}{k}"].to_numpy()
        rows.append(d)
        # key frames, front camera
        f_idx = int(m["videos/observation.images.front/file_index"])
        t0 = float(m["videos/observation.images.front/from_timestamp"])
        t1 = float(m["videos/observation.images.front/to_timestamp"])
        n = int(m.length)
        for key in KEYS:
            fi = int(round(key * (n - 1)))
            out = OUT_FRAMES / f"ep{ep}_front_f{fi}.jpg"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0 + (t1 - t0) * key:.4f}",
                            "-i", str(a.stage / f"videos/front/file-{f_idx:03d}.mp4"), "-frames:v", "1",
                            "-vf", "scale=320:240", str(out)], check=True)
        print(f"{label!r:45s} ep {ep:5d} len {n:4d} file {f_idx:03d}")
    df = pd.concat(rows, ignore_index=True)
    if a.gallery or a.selected:  # "robot 1, ep 581" on the first line, the wrapped task name below
        head = df.row_label.str.split("|").str[0] + ", ep " + df.episode_index.astype(str)
        df["row_label"] = head + "\n" + df.row_label.str.split("|").str[1]
    df.to_csv(out_csv, index=False, float_format="%.5f")
    print("wrote", out_csv, df.shape, "frames in", OUT_FRAMES)


if __name__ == "__main__":
    main()
