"""Turn a still character art (on a dark background) into a looping animated
Steam artwork showcase GIF: violet grade, pulsing glow on bright details,
"breathing" motion, rim light and a particle background.

Usage:
    python animate.py input.jpg --out showcase.gif
    python animate.py input.jpg --glow-box 240 385 240 480 --bg particles.mp4
"""
import argparse, os, subprocess, tempfile
import cv2
import numpy as np
from scipy import ndimage as nd
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
VIOLET = np.array([1.0, 0.35, 0.85])  # BGR


def cutout(im):
    """Alpha mask for a subject on a dark background."""
    lum = im.max(2)
    core = lum > 0.16
    lab, n = nd.label(core)
    sizes = nd.sum(core, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.where(sizes > 4000)[0])
    keep = nd.binary_fill_holes(nd.binary_closing(keep, iterations=6))
    zone = cv2.dilate(keep.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    soft = np.clip((lum - 0.07) / 0.12, 0, 1)
    a = np.where(keep, 1.0, np.where(zone, soft, 0.0)).astype(np.float32)
    return cv2.GaussianBlur(a, (3, 3), 0)


def violet_grade(rgb):
    bright = np.clip((rgb.mean(2) - 0.45) / 0.4, 0, 1)[..., None]
    return rgb * (1 - 0.35 * bright) + bright * 0.35 * np.array([1.0, 0.55, 0.85]) * rgb.max(2, keepdims=True)


def glow_mask(im, box):
    """Bright pinkish pixels; optionally limited to a box (y0 y1 x0 x1)."""
    lum = im.max(2)
    gm = ((lum > 0.62) & (im[..., 0] > im[..., 2] * 0.85)).astype(np.float32)
    if box:
        y0, y1, x0, x1 = box
        lim = np.zeros_like(gm); lim[y0:y1, x0:x1] = 1; gm *= lim
    return cv2.dilate(gm, np.ones((3, 3), np.uint8))


def particle_bg(n, w, h, fps, seed=7):
    """Procedural drifting violet streaks if no background video is given."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, (220, 2)) * [w, h]
    vel = rng.uniform(4, 14, (220, 1)) * np.array([[0.6, -0.8]])
    lens, wts = rng.uniform(3, 11, 220), rng.uniform(0.25, 1.0, 220)
    frames = []
    for f in range(n):
        part = np.zeros((h, w), np.float32)
        p = (pts + vel * f / fps) % [w, h]
        for (x, y), L, wt in zip(p, lens, wts):
            cv2.line(part, (int(x), int(y)), (int(x + L * 0.6), int(y - L * 0.8)), float(wt), 2, cv2.LINE_AA)
        bg = np.zeros((h, w, 3), np.float32) + (0.035, 0.012, 0.02)
        bg += (part[..., None] * 0.9 + cv2.GaussianBlur(part, (0, 0), 3)[..., None] * 1.0) * np.array([1.0, 0.25, 0.75])
        frames.append(bg)
    return frames


def video_bg(path, n, w, h, fps, x0, y0):
    raw = subprocess.run([FF, "-v", "error", "-i", path, "-vf", f"fps={fps},crop={w}:{h}:{x0}:{y0}",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="showcase.gif")
    ap.add_argument("--width", type=int, default=506)
    ap.add_argument("--height", type=int, default=824)
    ap.add_argument("--subject-height", type=int, default=600)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--glow-box", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--bg", help="background video (optional)")
    ap.add_argument("--bg-offset", type=int, nargs=2, default=(0, 0), metavar=("X", "Y"))
    ap.add_argument("--colors", type=int, default=96)
    a = ap.parse_args()

    W, H, S, FPS = a.width, a.height, a.subject_height, a.fps
    N = int(FPS * a.seconds)
    im = cv2.imread(a.input).astype(np.float32) / 255
    alpha, gm, rgb = cutout(im), glow_mask(im, a.glow_box), violet_grade(im)

    h0, w0 = alpha.shape
    sc = S / h0
    rs = lambda x: cv2.resize(x, (int(w0 * sc), S), interpolation=cv2.INTER_AREA)
    rgb, alpha, gm = rs(rgb), rs(alpha), rs(gm)
    if rgb.shape[1] >= W:  # center-crop to showcase width
        cx = (rgb.shape[1] - W) // 2
        rgb, alpha, gm = rgb[:, cx:cx + W], alpha[:, cx:cx + W], gm[:, cx:cx + W]
    gw = rgb.shape[1]; ox = (W - gw) // 2; oy = H - S
    rim = cv2.GaussianBlur(alpha, (0, 0), 14)

    bgs = video_bg(a.bg, N, W, H, FPS, *a.bg_offset) if a.bg else particle_bg(N, W, H, FPS)
    rng = np.random.default_rng(3)
    spikes = {int(f): rng.uniform(1.2, 1.9) for f in rng.choice(N, max(1, N // 6), replace=False)}

    tmp = tempfile.mkdtemp()
    for f in range(N):
        t = f / FPS
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 2.5)) + spikes.get(f, 0) * 0.6 + rng.uniform(-0.06, 0.06)
        M = cv2.getRotationMatrix2D((gw / 2, S), 0, 1 + 0.012 * np.sin(2 * np.pi * t / 5))  # breathing
        r2, a2, g2, rim2 = (cv2.warpAffine(x, M, (gw, S)) for x in (rgb, alpha, gm, rim))
        bloom = cv2.GaussianBlur(g2, (0, 0), 7) * pulse * 0.9 + cv2.GaussianBlur(g2, (0, 0), 2) * pulse * 0.5
        lit = r2 + g2[..., None] * (pulse - 0.6) * 0.8 + bloom[..., None] * VIOLET
        bg = bgs[f % len(bgs)].copy()
        reg = bg[oy:oy + S, ox:ox + gw]
        reg += rim2[..., None] * np.array([0.55, 0.08, 0.35]) * 0.5 + bloom[..., None] * VIOLET * 0.5
        reg[:] = reg * (1 - a2[..., None]) + lit * a2[..., None] + bloom[..., None] * VIOLET * 0.4 * (1 - a2[..., None])
        cv2.imwrite(os.path.join(tmp, f"{f:04d}.png"), (np.clip(bg, 0, 1) * 255).astype(np.uint8))

    pal = f"split[x][y];[x]palettegen=max_colors={a.colors}:stats_mode=full[p];[y][p]paletteuse=dither=none:diff_mode=rectangle"
    subprocess.run([FF, "-y", "-v", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
                    "-vf", pal, "-loop", "0", a.out], check=True)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
