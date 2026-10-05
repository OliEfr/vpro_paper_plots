#!/usr/bin/env python3
"""Screening aids for cherry-picking hardware rollouts (no success labels exist for the recordings).

    python screen.py last  <clips_dir> <out.jpg>   # grid of each episode's LAST frame (5 per row, 384 px wide)
    python screen.py strip <clips_dir> <ep> <out.jpg> [n]   # one episode, n evenly spaced frames at full size

The last frame tells success/failure for pick-and-place tasks (object in container or not); the
strip is for checking how a rollout got there before it goes in the video.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def last_frames(clips_dir, out, per_row=5, tile_w=384):
    files = sorted(Path(clips_dir).glob("ep*.mp4"))
    tiles = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(n - 2, 0))
        ok, fr = cap.read()
        if not ok:
            fr = np.zeros((240, 320, 3), np.uint8)
        h, w = fr.shape[:2]
        fr = cv2.resize(fr, (tile_w, int(tile_w * h / w)))
        cv2.rectangle(fr, (0, 0), (150, 36), (255, 255, 255), -1)
        cv2.putText(fr, f"{f.stem}  {n / 30:.0f}s", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(fr)
        cap.release()
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + per_row], axis=1) for i in range(0, len(tiles), per_row)]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def strip(clips_dir, ep, out, n_frames=8):
    cap = cv2.VideoCapture(str(Path(clips_dir) / f"ep{int(ep):03d}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, n - 1, n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        cv2.putText(fr, f"{i / 30:.1f}s", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        tiles.append(fr)
    half = (len(tiles) + 1) // 2
    rows = [np.concatenate(tiles[:half], 1), np.concatenate(tiles[half:half * 2] if len(tiles) % 2 == 0 else tiles[half:] + [np.full_like(tiles[0], 255)], 1)]
    cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    if sys.argv[1] == "last":
        last_frames(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "strip":
        strip(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 8)
