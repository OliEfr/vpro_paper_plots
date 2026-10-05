#!/usr/bin/env python3
"""Cut per-episode clips out of LeRobot v3.0 datasets, and render contact sheets for screening.

LeRobot v3 stores many episodes concatenated in one mp4 per chunk/file; `meta/episodes/*.parquet`
holds, per episode and video key, the file index and the from/to timestamps. This script turns
those into one H.264 mp4 per episode (re-encoded, so cuts are frame-accurate and AV1 sources
become something every player opens).

    python cut.py episode <dataset_root> <video_key> <episode_index> <out.mp4>
    python cut.py all     <dataset_root> <video_key> <out_dir>            # every episode
    python cut.py sheet   <dataset_root> <clips_dir> <out.jpg>            # contact sheet from `all` clips: 1 row / episode
    python cut.py tasks   <dataset_root>                                  # episode -> task table

`dataset_root` is a local LeRobot dataset directory (meta/ + videos/), e.g. video/raw/<repo>.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# The conda ffmpeg on st-07 has no libx264; the Ubuntu one has libx264 + dav1d.
FFMPEG, FFPROBE = "/usr/bin/ffmpeg", "/usr/bin/ffprobe"


def episodes(root):
    files = sorted(glob.glob(str(Path(root) / "meta" / "episodes" / "*" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)


def tasks(root):
    p = Path(root) / "meta" / "tasks.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return {int(v): k for k, v in df["task_index"].items()}
    return {}


def info(root):
    return json.load(open(Path(root) / "meta" / "info.json"))


def video_path(root, key, chunk, file_index):
    tpl = info(root)["video_path"]
    return Path(root) / tpl.format(video_key=key, chunk_index=int(chunk), file_index=int(file_index))


def episode_span(root, key, ep):
    df = episodes(root)
    row = df[df.episode_index == ep].iloc[0]
    src = video_path(root, key, row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"])
    return src, float(row[f"videos/{key}/from_timestamp"]), float(row[f"videos/{key}/to_timestamp"]), row


def codec(path):
    return subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()


def decoder_args(path):
    # ffmpeg's default AV1 path tries hardware decoding and fails on this box; dav1d is built in.
    return ["-c:v", "libdav1d"] if codec(path) == "av1" else []


def cut_episode(root, key, ep, out):
    src, t0, t1, _ = episode_span(root, key, ep)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # -ss after -i = accurate seek (decode from start of the file); fine for files of a few hundred MB.
    subprocess.run([FFMPEG, "-v", "error", "-y", *decoder_args(src), "-i", str(src), "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}",
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def cut_all(root, key, out_dir):
    df = episodes(root)
    for ep in df.episode_index:
        cut_episode(root, key, int(ep), Path(out_dir) / f"ep{int(ep):03d}.mp4")
        print("cut", ep, flush=True)


def contact_sheet(root, clips_dir, out, n_frames=5, tile=192):
    """One row per episode (from the per-episode H.264 clips of `cut all`), n_frames evenly spaced
    frames, episode index + duration + task on the left. OpenCV cannot decode the AV1 sources, so
    the sheet reads the re-encoded clips, not the raw dataset videos."""
    df = episodes(root)
    rows = []
    for _, r in df.iterrows():
        ep = int(r.episode_index)
        cap = cv2.VideoCapture(str(Path(clips_dir) / f"ep{ep:03d}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tiles = []
        for idx in np.linspace(0, max(n - 1, 0), n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if not ok:
                f = np.zeros((tile, tile, 3), np.uint8)
            h, w = f.shape[:2]
            tiles.append(cv2.resize(f, (int(tile * w / h), tile)))
        cap.release()
        lab = np.full((tile, 260, 3), 255, np.uint8)
        task = r["tasks"][0] if hasattr(r["tasks"], "__len__") and not isinstance(r["tasks"], str) else str(r["tasks"])
        cv2.putText(lab, f"ep {ep}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(lab, f"{n / fps:.1f}s", (8, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1, cv2.LINE_AA)
        for k, chunk_s in enumerate([task[i:i + 22] for i in range(0, min(len(task), 66), 22)]):
            cv2.putText(lab, chunk_s, (8, 115 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        rows.append((ep, np.concatenate([lab] + tiles, axis=1)))
    rows.sort(key=lambda x: x[0])
    wmax = max(r.shape[1] for _, r in rows)
    rows = [np.pad(r, ((0, 4), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for _, r in rows]
    sheet = np.concatenate(rows, axis=0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def task_table(root):
    df = episodes(root)
    for _, r in df.iterrows():
        print(int(r.episode_index), int(r.length), r["tasks"])


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "episode":
        cut_episode(args[0], args[1], int(args[2]), args[3])
    elif cmd == "all":
        cut_all(args[0], args[1], args[2])
    elif cmd == "sheet":
        contact_sheet(args[0], args[1], args[2])
    elif cmd == "tasks":
        task_table(args[0])
    else:
        sys.exit(__doc__)
