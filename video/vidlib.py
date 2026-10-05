"""Shared helpers for the paper companion video: clip loading, white-canvas layout, text, ffmpeg output.

Everything here is plain OpenCV + Pillow + ffmpeg -- no GPU, no model code. Style is deliberately
minimal: white background, black DejaVu Sans, grey secondary text, nothing animated.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FFMPEG = "/usr/bin/ffmpeg"  # the conda ffmpeg lacks libx264
W, H = 1920, 1080
FPS = 30
WHITE = (255, 255, 255)
INK = (20, 20, 20)
MUTED = (110, 110, 110)
PLACEHOLDER_BG = (236, 236, 236)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

_fonts = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size)
    return _fonts[key]


# ----------------------------------------------------------------------------- text
def draw_text(img, s, xy, size=36, color=INK, bold=False, anchor="la", max_width=None):
    """Draw text on an RGB uint8 array in place. anchor follows Pillow ("la" = left/ascender,
    "ma" = centre/ascender, "ra" = right/ascender). Wraps to max_width (px) if given."""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    f = font(size, bold)
    lines = [s]
    if max_width:
        lines, cur = [], ""
        for w in s.split():
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) > max_width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
    x, y = xy
    for ln in lines:
        d.text((x, y), ln, font=f, fill=color, anchor=anchor)
        y += int(size * 1.25)
    img[:] = np.asarray(pil)
    return y


def text_width(s, size=36, bold=False):
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(s, font=font(size, bold))


def blank():
    return np.full((H, W, 3), 255, np.uint8)


# ----------------------------------------------------------------------------- clips
def read_frames(path, t0=None, t1=None):
    """All frames of a video (RGB uint8), optionally restricted to [t0, t1) seconds."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if t0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t0 * fps)))
    n_max = None if t1 is None else int(round((t1 - (t0 or 0)) * fps))
    out = []
    while True:
        if n_max is not None and len(out) >= n_max:
            break
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out, fps


class Clip:
    """A sequence of frames with a source fps, played back at `speed` on the FPS canvas.
    Freezes on its last frame once finished."""

    def __init__(self, frames, fps, speed=1.0, label=None, hold=0.0, crop=None, loop=False):
        if crop:  # (x0, y0, x1, y1) in source pixels
            x0, y0, x1, y1 = crop
            frames = [f[y0:y1, x0:x1] for f in frames]
        self.frames, self.fps, self.speed, self.label, self.hold = frames, fps, speed, label, hold
        self.loop = loop
        self.placeholder = None

    @classmethod
    def from_file(cls, path, speed=1.0, label=None, t0=None, t1=None, hold=0.0, crop=None, loop=False, fps=None):
        frames, src_fps = read_frames(path, t0, t1)
        return cls(frames, fps or src_fps, speed, label, hold, crop, loop)

    @classmethod
    def sequence(cls, clips, gap=0.5, loop=True):
        """Several clips (same fps, same frame size) played one after another, with `gap` seconds of
        the last frame held between them; loops by default."""
        fps = clips[0].fps
        frames = []
        for c in clips:
            frames += c.frames
            frames += [c.frames[-1]] * int(gap * fps)
        return cls(frames, fps, clips[0].speed, clips[0].label, 0.0, None, loop)

    @classmethod
    def placeholder_clip(cls, text, size=(256, 256), seconds=4.0, label=None):
        img = np.full((size[1], size[0], 3), PLACEHOLDER_BG, np.uint8)
        draw_text(img, "placeholder", (size[0] // 2, size[1] // 2 - 30), size=max(14, size[0] // 14), color=INK, anchor="mm")
        draw_text(img, text, (size[0] // 2, size[1] // 2 + 10), size=max(12, size[0] // 18), color=INK, anchor="mm", max_width=size[0] - 20)
        c = cls([img], 1.0, 1.0, label)
        c.placeholder = seconds
        return c

    @property
    def duration(self):
        """Seconds on the canvas."""
        if self.placeholder:
            return self.placeholder
        return len(self.frames) / self.fps / self.speed + self.hold

    def frame_at(self, t):
        if self.placeholder:
            return self.frames[0]
        i = int(t * self.speed * self.fps)
        if self.loop:
            i %= len(self.frames)
        return self.frames[min(i, len(self.frames) - 1)]


def fit(img, box_w, box_h):
    """Scale img to fit a box, keep aspect, return (resized, w, h)."""
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp), nw, nh


def paste(canvas, img, x, y):
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


# ----------------------------------------------------------------------------- output
class Writer:
    """Pipes RGB frames into ffmpeg -> H.264 yuv420p mp4 at FPS."""

    def __init__(self, path, crf=18):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.p = subprocess.Popen(
            [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", self.path],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        assert frame.shape == (H, W, 3), frame.shape
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        if self.p.returncode:
            raise RuntimeError(f"ffmpeg failed for {self.path}")
        return self.n / FPS


def concat(paths, out):
    """Lossless-ish concat of same-format mp4s via the concat demuxer."""
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    lst.unlink()


# ----------------------------------------------------------------------------- segments
def title_card(writer, title, subtitle=None, seconds=2.0):
    img = blank()
    draw_text(img, title, (W // 2, H // 2 - (40 if subtitle else 0)), size=80, bold=True, anchor="mm")
    if subtitle:
        draw_text(img, subtitle, (W // 2, H // 2 + 50), size=40, color=INK, anchor="mm")
    for _ in range(int(seconds * FPS)):
        writer.write(img)


def white_gap(writer, seconds=0.5):
    img = blank()
    for _ in range(int(seconds * FPS)):
        writer.write(img)


def speed_tag(speed):
    return f"{speed:g}x speed" if speed != 1 else "Real time"


def layout_rows(rows, top, bottom, tile_gap=24, label_h=0, subtitle_h=0, label_above=False):
    """Compute tile boxes for a list of rows (each row: (n tiles, aspect w/h)). Returns per row:
    (y_subtitle, [(x, y, w, h) tile boxes], y_label). Rows share equal height; tiles keep the
    source aspect and are as large as the row allows."""
    n_rows = len(rows)
    row_gap = 40
    avail_h = bottom - top - row_gap * (n_rows - 1)
    row_h = avail_h // n_rows
    tile_h_max = row_h - subtitle_h - label_h
    out = []
    y = top
    for n, aspect in rows:
        tile_w_max = (W - 160 - tile_gap * (n - 1)) // n
        tile_w = min(tile_w_max, int(tile_h_max * aspect))
        tile_h = min(tile_h_max, int(tile_w / aspect))
        total = n * tile_w + (n - 1) * tile_gap
        x0 = (W - total) // 2
        # centre the (possibly shorter) tile block vertically in the row's tile area
        y_tiles = y + subtitle_h + (label_h if label_above else 0) + (tile_h_max - tile_h) // 2
        boxes = [(x0 + i * (tile_w + tile_gap), y_tiles, tile_w, tile_h) for i in range(n)]
        y_lab = (y_tiles - label_h) if label_above else (y_tiles + tile_h)
        out.append((y + 0, boxes, y_lab))
        y += row_h + row_gap
    return out


def draw_title(img, title, size=64):
    """Bold centred title; shrinks until it fits the canvas width with margins."""
    while size > 36 and text_width(title, size, bold=True) > W - 240:
        size -= 4
    draw_text(img, title, (W // 2, 50), size=size, bold=True, anchor="ma")


def draw_note(img, note):
    draw_text(img, note, (W - 60, H - 48), size=24, color=MUTED, anchor="ra")


def render_rows(writer, title, rows, seconds=None, note=None, title_size=64, sub_size=34, label_size=26, label_above=False):
    """rows: list of dicts {subtitle, clips: [Clip], speed_note (optional str)}.
    All clips start together; segment lasts until the longest clip is done (or `seconds`)."""
    top = 150
    bottom = H - 70
    any_sub = any(r.get("subtitle") for r in rows)
    any_label = any(c.label for r in rows for c in r["clips"]) or any(r.get("caption") for r in rows)

    def aspect(r):
        f = r["clips"][0].frame_at(0)
        return f.shape[1] / f.shape[0]

    geo = layout_rows([(len(r["clips"]), aspect(r)) for r in rows], top, bottom,
                      subtitle_h=int(sub_size * 1.9) if any_sub else 0,
                      label_h=int(label_size * (1.9 if label_above else 2.6)) if any_label else 0, label_above=label_above)
    dur = seconds or max(c.duration for r in rows for c in r["clips"])
    n_frames = int(round(dur * FPS))
    base = blank()
    draw_title(base, title, title_size)
    if note:
        draw_note(base, note)
    for r, (y_sub, boxes, y_lab) in zip(rows, geo):
        if r.get("subtitle"):
            s = r["subtitle"]
            if r.get("speed_note"):
                s += f"   ({r['speed_note']})"
            draw_text(base, s, (W // 2, y_sub), size=sub_size, color=INK, anchor="ma")
        if r.get("caption"):  # one centred caption under the whole row instead of per-tile labels
            draw_text(base, r["caption"], (W // 2, y_lab + 8), size=label_size, color=INK, anchor="ma")
        for c, (x, y, w, h) in zip(r["clips"], boxes):
            if c.label and not r.get("caption"):
                draw_text(base, c.label, (x + w // 2, y_lab + 8), size=label_size, color=INK,
                          anchor="ma", max_width=w + 20)
    for i in range(n_frames):
        t = i / FPS
        img = base.copy()
        for r, (y_sub, boxes, y_lab) in zip(rows, geo):
            for c, (x, y, w, h) in zip(r["clips"], boxes):
                f, nw, nh = fit(c.frame_at(t), w, h)
                paste(img, f, x + (w - nw) // 2, y + (h - nh) // 2)
        writer.write(img)
    return dur


def draw_arrow(img, x0, x1, y, thick=4):
    """Right-pointing arrow from x0 to x1 at height y (RGB array, in place)."""
    cv2.line(img, (x0, y), (x1 - 14, y), INK, thick, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[x1, y], [x1 - 22, y - 12], [x1 - 22, y + 12]])], INK, cv2.LINE_AA)


def draw_vertical_text(img, s, x_center, y_center, size=26, color=INK, bold=False, up=True):
    """Text rotated 90 degrees (reading bottom-to-top if up=True), centred on (x_center, y_center)."""
    f = font(size, bold)
    tw = int(text_width(s, size, bold)) + 8
    th = int(size * 1.4)
    tmp = Image.new("RGB", (tw, th), (255, 255, 255))
    ImageDraw.Draw(tmp).text((4, 0), s, font=f, fill=color)
    arr = np.asarray(tmp)
    arr = np.rot90(arr, 1 if up else 3)
    h, w = arr.shape[:2]
    x0, y0 = int(x_center - w // 2), int(y_center - h // 2)
    mask = arr.sum(2) < 740
    region = img[y0:y0 + h, x0:x0 + w]
    region[mask[:region.shape[0], :region.shape[1]]] = arr[:region.shape[0], :region.shape[1]][mask[:region.shape[0], :region.shape[1]]]


def draw_down_arrow(img, x, y0, y1, thick=4):
    cv2.line(img, (x, y0), (x, y1 - 14), INK, thick, cv2.LINE_AA)
    cv2.fillPoly(img, [np.array([[x, y1], [x - 12, y1 - 22], [x + 12, y1 - 22]])], INK, cv2.LINE_AA)


def render_canvas(writer, title, seconds, paint, note=None, subtitle=None):
    """Generic segment: fixed title (+ optional subtitle + note); `paint(img, t)` draws the body
    onto a copy of that base for canvas time t."""
    base = blank()
    draw_title(base, title)
    if subtitle:
        draw_text(base, subtitle, (W // 2, 150), size=36, anchor="ma", max_width=W - 300)
    if note:
        draw_note(base, note)
    for i in range(int(round(seconds * FPS))):
        img = base.copy()
        paint(img, i / FPS)
        writer.write(img)
    return seconds


def render_full_clip(writer, title, clip, subtitle=None, note=None, pad_top=150):
    """One existing video, scaled to fit under a fixed title + optional subtitle."""
    base = blank()
    draw_title(base, title)
    y0 = pad_top
    if subtitle:
        draw_text(base, subtitle, (W // 2, y0), size=36, anchor="ma", max_width=W - 300)
        y0 += 70
    if note:
        draw_note(base, note)
    box_h = H - y0 - 70
    box_w = W - 160
    n = int(round(clip.duration * FPS))
    for i in range(n):
        img = base.copy()
        f, nw, nh = fit(clip.frame_at(i / FPS), box_w, box_h)
        paste(img, f, (W - nw) // 2, y0 + (box_h - nh) // 2)
        writer.write(img)
    return clip.duration
