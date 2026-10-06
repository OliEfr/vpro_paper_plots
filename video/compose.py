#!/usr/bin/env python3
"""Build the paper companion video from `sources.yaml`.

    python compose.py            # all slides -> slides/slide*.mp4 + slides/paper_video.mp4
    python compose.py 4 6        # only slides 4 and 6 (+ re-concat)

Every clip the video shows is listed in sources.yaml with its origin (repo id / path, episode).
Clips are the per-episode H.264 files produced by cut.py (or existing mp4s under figures/ and
vpro_mimicgen_eval). A `placeholder:` entry renders a grey box with the text instead of footage,
for material that could not be fetched yet (MN5 was down when this was built).

Conventions (from Oliver, 2026-10-05): white background, black text, no narration; simulation at
2x, hardware in real time, the speed tag bottom-right in grey; front camera for hardware,
frontview for sim demos; tiles loop while a slide plays instead of waiting for the longest clip.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import vidlib as V

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "slides"
SPEC = yaml.safe_load(open(HERE / "sources.yaml"))


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def clip(entry, speed=1.0, size=(256, 256), loop=False):
    """sources.yaml clip entry -> vidlib.Clip."""
    if "placeholder" in entry:
        return V.Clip.placeholder_clip(entry["placeholder"], size=size, label=entry.get("label"))
    return V.Clip.from_file(resolve(entry["clip"]), speed=entry.get("speed", speed), label=entry.get("label"),
                            t0=entry.get("t0"), t1=entry.get("t1"), hold=entry.get("hold", 0.0),
                            crop=tuple(entry["crop"]) if entry.get("crop") else None, loop=entry.get("loop", loop),
                            fps=entry.get("fps"))


def card(w, s):
    if s.get("card", True):
        V.title_card(w, s.get("card_title", s["title"]), s.get("card_subtitle"))


# ----------------------------------------------------------------------------- slides
def slide_sim(w, s):
    """Slides 1 and 2: cross-embodiment demos (top) + deployment on the target robot (bottom).
    Every tile loops; the slide runs for `seconds` (default: the longest single clip)."""
    card(w, s)
    speed = s.get("speed", 2.0)
    rows = []
    for r in s["rows"]:
        rows.append({"subtitle": r.get("subtitle"), "caption": r.get("caption"),
                     "clips": [clip(c, speed, loop=True) for c in r["clips"]]})
    seconds = s.get("seconds") or max(c.duration for r in rows for c in r["clips"])
    V.render_rows(w, s["title"], rows, seconds=seconds, note=V.speed_tag(speed))


def slide_grid(w, s):
    """Slides 4 and 6: 2x3 grid of tasks; each tile cycles through its own rollouts and loops.
    The slide runs until the tile with the most footage has shown everything once."""
    card(w, s)
    cells = s["cells"]
    tiles = []
    for c in cells:
        seq = V.Clip.sequence([clip(e) for e in c["clips"]], gap=0.5, loop=True)
        seq.label = c["task"]
        tiles.append(seq)
    seconds = (s.get("seconds") or max(t.duration for t in tiles)) + s.get("extra_seconds", 0)
    rows = [{"subtitle": None, "clips": tiles[:3]}, {"subtitle": None, "clips": tiles[3:6]}]
    V.render_rows(w, s["title"], rows, note=V.speed_tag(1.0), seconds=seconds, label_size=28, label_above=True)


def slide_transfer(w, s):
    """Slide 5: fixed title, then one sub-slide per part. Part kinds:
    - arrow:     from scratch: source demo -> arrow -> model outputs (+ reference), panels cropped from a clip
    - existing:  show an existing clip as is (optionally a time window)
    - pairlat:   from scratch: robot / human clips of the same movement + their latent heatmaps
    - gridtiles: from scratch: latent grid rebuilt tile by tile, text left, transfer arrow right
    """
    card(w, s)
    for part in s["parts"]:
        kind = part.get("kind", "existing")
        for e in part["clips"]:
            if kind == "arrow":
                arrow_slide(w, s["title"], part["subtitle"], e)
            elif kind == "pairlat":
                pairlat(w, s["title"], part["subtitle"], e)
            elif kind == "gridtiles":
                gridtiles(w, s["title"], part["subtitle"], e)
            else:
                V.render_full_clip(w, s["title"], clip(e), subtitle=part["subtitle"], note=e.get("note"))


def arrow_slide(w, title, subtitle, e):
    """Source demo -> arrow (Transfer / Latent) -> model outputs (and optionally a real reference).
    Every panel is cropped out of an existing xemb clip; labels are redrawn in the video's style."""
    base = clip(e)
    y0, y1 = e["panel_y"]
    aspect = e.get("aspect")                        # source clips that squashed the 4:3 camera into squares get it back

    def crop(f, x0, x1):
        p = f[y0:y1, x0:x1]
        if aspect:
            p = cv2.resize(p, (int(round(p.shape[0] * aspect)), p.shape[0]), interpolation=cv2.INTER_CUBIC)
        return p

    panels = [V.Clip([crop(f, x0, x1) for f in base.frames], base.fps) for x0, x1 in [e["src"]] + [tuple(o) for o in e["outputs"]]]
    labels = [e["src_label"]] + list(e["out_labels"])
    n_out = len(panels) - 1
    gap, arrow_w = 40, 120
    tile = min(560, (V.W - 120 - 2 * gap - arrow_w - gap * (n_out - 1)) // (n_out + 1))
    f0 = panels[0].frame_at(0)
    tile_h = int(tile * f0.shape[0] / f0.shape[1]) if f0.shape[1] >= f0.shape[0] else tile
    total = tile * (n_out + 1) + 2 * gap + arrow_w + gap * (n_out - 1)
    xs = [(V.W - total) // 2]
    xs.append(xs[0] + tile + gap + arrow_w + gap)
    for k in range(1, n_out):
        xs.append(xs[-1] + tile + gap)
    y = 150 + (V.H - 70 - 150 - tile_h - 40) // 2

    def paint(img, t):
        for c, x in zip(panels, xs):
            f, nw, nh = V.fit(c.frame_at(t), tile, tile_h)
            V.paste(img, f, x + (tile - nw) // 2, y + (tile_h - nh) // 2)
        ax0, ax1 = xs[0] + tile + gap, xs[1] - gap
        V.draw_arrow(img, ax0, ax1, y + tile_h // 2)
        V.draw_text(img, "Transfer", ((ax0 + ax1) // 2, y + tile_h // 2 - 44), size=26, color=V.INK, anchor="ma")
        V.draw_text(img, "Latent", ((ax0 + ax1) // 2, y + tile_h // 2 + 16), size=26, color=V.INK, anchor="ma")
        for lab, x in zip(labels, xs):
            V.draw_text(img, lab, (x + tile // 2, y + tile_h + 16), size=26, color=V.INK, anchor="ma", max_width=tile)

    V.render_canvas(w, title, base.duration, paint, note=e.get("note"), subtitle=subtitle)


def pairlat(w, title, subtitle, e):
    """Robot and human clip of the same movement (front camera), one task caption centred under both,
    and below that the latent sequences the LAM produces for both (8 dims over the 1 s window,
    embodiment mean removed) with its caption above. Panels are cropped from the existing
    realworld-transfer clip, inset to drop its blue frame and burnt-in titles."""
    base = clip(e)
    (vx0, vx1), (vx2, vx3), (vy0, vy1) = e["robot_x"], e["human_x"], e["video_y"]
    (hx0, hx1), (hy0, hy1) = e["heat_x"], e["heat_y"]
    reps = e.get("repeat", 1)                # play the example this many times
    rob = V.Clip([f[vy0:vy1, vx0:vx1] for f in base.frames], base.fps, loop=True)
    hum = V.Clip([f[vy0:vy1, vx2:vx3] for f in base.frames], base.fps, loop=True)
    heat = V.Clip([f[hy0:hy1, hx0:hx1] for f in base.frames], base.fps, loop=True)
    tile_w, tile_h = 760, 540
    gap = 60
    xs = [(V.W - 2 * tile_w - gap) // 2, (V.W - 2 * tile_w - gap) // 2 + tile_w + gap]
    y = 225
    _, _, vh = V.fit(rob.frame_at(0), tile_w, tile_h)
    y_cap = y + vh + 16                      # task caption under both videos
    y_heat_lab = y_cap + 60                  # heatmap caption, above the heatmap
    y_heat = y_heat_lab + 40

    def paint(img, t):
        for c, x in zip((rob, hum), xs):
            f, nw, nh = V.fit(c.frame_at(t), tile_w, tile_h)
            V.paste(img, f, x + (tile_w - nw) // 2, y)
        V.draw_text(img, e["task"], (V.W // 2, y_cap), size=26, color=V.INK, anchor="ma")
        V.draw_text(img, e.get("heat_label", "Latent dimensions over the movement"), (V.W // 2, y_heat_lab), size=26, color=V.INK, anchor="ma")
        f, nw, nh = V.fit(heat.frame_at(t), 2 * tile_w + gap, V.H - 70 - y_heat)
        V.paste(img, f, (V.W - nw) // 2, y_heat)

    V.render_canvas(w, title, base.duration * reps, paint, note=e.get("note"), subtitle=subtitle)


def gridtiles(w, title, subtitle, e):
    """Latent-action grid rebuilt from the individual tiles of xemb_latent_grid_evaltasks.mp4
    (7 columns x 5 rows of 200 px tiles): row 0 = source demos (blue frame, 'Source motion'),
    column 0 = start frames (orange frame, 'Scene'), the rest = LAM roll-outs. Movement-direction
    labels on top, a 'Transfer latent' arrow on the right, explanatory text on the left. The source
    tiles are the 4:3 camera squashed to squares; `aspect` stretches them back."""
    base = clip(e)
    cols, rows, ts = e["cols_x"], e["rows_y"], e["tile_px"]
    aspect = e.get("aspect", 1.0)
    TW = e.get("tile", 150)
    TH = int(round(TW / aspect))
    tiles = {}
    for r, yy in enumerate(rows):
        for c, xx in enumerate(cols):
            if r == 0 and c == 0:
                continue
            tiles[(r, c)] = V.Clip([cv2.resize(f[yy:yy + ts, xx:xx + ts], (TW, TH), interpolation=cv2.INTER_AREA) for f in base.frames], base.fps, loop=True)
    for a, b in e.get("swap_source", []):           # source-row tiles whose columns are exchanged
        tiles[(0, a)], tiles[(0, b)] = tiles[(0, b)], tiles[(0, a)]
    reps = e.get("repeat", 1)
    g = 10
    frame_pad = 7
    top, bottom = 150, V.H - 70
    grid_h = 5 * TH + 4 * g
    grid_w = 7 * TW + 6 * g
    label_h = 90                                     # axis label + direction labels
    gy = top + label_h + (bottom - top - label_h - grid_h) // 2
    gx = V.W - 60 - 110 - grid_w                     # room for the arrow column on the right
    BLUE, ORANGE = (0, 90, 181), (230, 120, 0)
    text_w = gx - 60 - 60 - 40

    def paint(img, t):
        for (r, c), cl in tiles.items():
            V.paste(img, cl.frame_at(t), gx + c * (TW + g), gy + r * (TH + g))
        # blue frame around the source row (columns 1..6) + vertical two-line label to its left
        x0, y0 = gx + (TW + g) - frame_pad, gy - frame_pad
        x1, y1 = gx + grid_w + frame_pad, gy + TH + frame_pad
        cv2.rectangle(img, (x0, y0), (x1, y1), BLUE, 2)
        V.draw_vertical_text(img, "Source", x0 - 44, gy + TH // 2, size=22, color=BLUE)
        V.draw_vertical_text(img, "motion", x0 - 18, gy + TH // 2, size=22, color=BLUE)
        # orange frame around the scene column (rows 1..4) + label above
        x0, y0 = gx - frame_pad, gy + (TH + g) - frame_pad
        x1, y1 = gx + TW + frame_pad, gy + grid_h + frame_pad
        cv2.rectangle(img, (x0, y0), (x1, y1), ORANGE, 2)
        V.draw_text(img, e.get("scene_label", "Scene"), (gx + TW // 2, gy + (TH + g) - frame_pad - 32), size=22, color=ORANGE, anchor="ma")
        # direction labels + axis label on top
        for c, lab in enumerate(e["directions"], start=1):
            V.draw_text(img, lab, (gx + c * (TW + g) + TW // 2, gy - frame_pad - 44), size=26, color=V.INK, anchor="ma")
        V.draw_text(img, "Movement direction", (gx + (TW + g) + (grid_w - (TW + g)) // 2, gy - frame_pad - 84), size=26, color=V.INK, bold=True, anchor="ma")
        # transfer arrow on the right
        ax = gx + grid_w + 34
        V.draw_down_arrow(img, ax, gy + (TH + g), gy + grid_h)
        V.draw_vertical_text(img, "Transfer latent", ax + 30, gy + (TH + g) + (grid_h - (TH + g)) // 2, size=26, color=V.INK, up=False)
        # text column
        yy = top + 40
        yy = V.draw_text(img, subtitle, (60, yy), size=40, bold=True, max_width=text_w) + 24
        for para in e.get("text", []):
            yy = V.draw_text(img, para, (60, yy), size=28, color=V.INK, max_width=text_w) + 18

    V.render_canvas(w, title, base.duration * reps, paint, note=e.get("note"))


BUILDERS = {"sim": slide_sim, "grid": slide_grid, "transfer": slide_transfer}


def build(nums):
    OUT.mkdir(exist_ok=True)
    for n in nums:
        s = SPEC["slides"][n]
        out = OUT / f"slide{n}_{s['slug']}.mp4"
        w = V.Writer(out)
        BUILDERS[s["kind"]](w, s)
        print(f"slide {n}: {w.close():.1f} s -> {out.name}", flush=True)


def final():
    parts = [OUT / f"slide{n}_{s['slug']}.mp4" for n, s in sorted(SPEC["slides"].items())]
    missing = [p.name for p in parts if not p.exists()]
    if missing:
        print("not concatenating, missing:", missing)
        return
    stale = [p for p in OUT.glob("slide*.mp4") if p not in parts]
    for p in stale:
        p.unlink()
        print("removed stale", p.name)
    V.concat(parts, OUT / "paper_video.mp4")
    print("-> paper_video.mp4")


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or sorted(SPEC["slides"])
    build(nums)
    final()
