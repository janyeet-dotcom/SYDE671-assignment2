"""
Part 2.2: hybrid images (Oliva, Torralba & Schyns, SIGGRAPH 2006).

    low    = G_low * image_a             (low-pass: seen from far away)
    high   = image_b - G_high * image_b  (impulse minus Gaussian: seen up close)
    hybrid = low + high

Pipeline: align on two reference points (e.g. the eyes) -> filter -> add.

Usage: python simple_2_2_hybrid.py LOW_IMAGE HIGH_IMAGE [--sigma-low S] [--sigma-high S]
           [--pts-low x1 y1 x2 y2] [--pts-high x1 y1 x2 y2] [--no-align]
  Omit --pts-* to click the two points yourself (needs a display).
  --no-align: pair is already aligned, just resize the high image to the low one.
  Cutoff frequency is about 1 / (2 pi sigma) cycles per pixel.
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

CANVAS_W, CANVAS_H = 520, 640   # both images are warped onto this canvas
REF_CENTER, REF_DIST = (260, 250), 150  # where the two reference points land


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def pick_points(img, title):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    pts = plt.ginput(2, timeout=0)
    plt.close()
    return np.float32(pts)


def align(img, pts):
    """Rotate/scale/translate so the two points land on fixed horizontal positions."""
    cx, cy = REF_CENTER
    target = np.float32([[cx - REF_DIST / 2, cy], [cx + REF_DIST / 2, cy]])
    M, _ = cv2.estimateAffinePartial2D(pts, target, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def blur(img, sigma):
    """Gaussian low-pass with mirrored borders (gray or colour)."""
    if img.ndim == 3:
        return np.dstack([blur(img[..., c], sigma) for c in range(3)])
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    p = ksize // 2
    return fftconvolve(np.pad(img, p, mode="reflect"), g @ g.T, mode="valid").astype(np.float32)


def hybrid(low_img, high_img, s_low, s_high):
    low = blur(low_img, s_low)
    high = high_img - blur(high_img, s_high)
    return low, high, np.clip(low + high, 0, 1)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to3(a):
    return a if a.ndim == 3 else np.dstack([a] * 3)


def log_spectrum(g):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)


def shrunk(img, f):
    """img at 1/f size, centred on a white canvas of the original size (stepping back)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    canvas = np.ones_like(img)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def panels(items, shape, size, name):
    """items: list of (image, title[, cmap]); saves a grid figure."""
    fig, axes = plt.subplots(*shape, figsize=size)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (im, title, *cmap) in zip(np.ravel(axes), items):
        ax.imshow(np.clip(im, 0, 1), cmap=cmap[0] if cmap else ("gray" if im.ndim == 2 else None),
                  vmin=0 if im.ndim == 2 else None, vmax=1 if im.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(name, dpi=100)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("low_image")
    p.add_argument("high_image")
    p.add_argument("--sigma-low", type=float, default=7.0)
    p.add_argument("--sigma-high", type=float, default=3.5)
    p.add_argument("--pts-low", type=float, nargs=4)
    p.add_argument("--pts-high", type=float, nargs=4)
    p.add_argument("--no-align", action="store_true")
    args = p.parse_args()
    sl, sh = args.sigma_low, args.sigma_high

    low_rgb, high_rgb = load_rgb(args.low_image), load_rgb(args.high_image)

    # 1) Align
    if args.no_align:
        h, w = low_rgb.shape[:2]
        low_al = low_rgb
        high_al = cv2.resize(high_rgb, (w, h), interpolation=cv2.INTER_AREA)
        panels([(low_al, "Low-frequency image"), (high_al, "High-frequency image")],
               (1, 2), (12, 6), "part2_2_alignment.png")
    else:
        pl = (np.float32(args.pts_low).reshape(2, 2) if args.pts_low
              else pick_points(low_rgb, "LOW image: click the two reference points"))
        ph = (np.float32(args.pts_high).reshape(2, 2) if args.pts_high
              else pick_points(high_rgb, "HIGH image: click the same two points"))
        print("Reuse with: --pts-low", *pl.ravel().round(1), "--pts-high", *ph.ravel().round(1))
        low_al, high_al = align(low_rgb, pl), align(high_rgb, ph)
        fig = panels([(low_rgb, "Low image (original)"), (low_al, "aligned"),
                      (high_rgb, "High image (original)"), (high_al, "aligned")],
                     (1, 4), (18, 6), "part2_2_alignment.png")
        for ax, pts in zip(fig.axes[::2], (pl, ph)):
            ax.plot(*pts.T, "r+", ms=14, mew=2)
        fig.savefig("part2_2_alignment.png", dpi=100)

    # 2) Grayscale hybrid with frequency analysis
    g_low, g_high = gray(low_al), gray(high_al)
    low, high, hyb = hybrid(g_low, g_high, sl, sh)
    high_vis = 0.5 + 2 * high
    cols = [(g_low, "Low image"), (g_high, "High image"), (low, f"Low-passed (sigma={sl:g})"),
            (high_vis, f"High-passed (sigma={sh:g})"), (hyb, "Hybrid")]
    spectra = [log_spectrum(x) for x in (g_low, g_high, low, high, hyb)]
    items = [(im, t) for im, t in cols] + [(s, "log |FFT|", "gray") for s in spectra]
    # imshow of spectra must not be clipped to [0, 1]: draw them separately
    fig, axes = plt.subplots(2, 5, figsize=(22, 10.5))
    for ax in axes.flat:
        ax.axis("off")
    for j, (im, t) in enumerate(cols):
        axes[0, j].imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(t, fontsize=10)
        axes[1, j].imshow(spectra[j], cmap="gray")
        axes[1, j].set_title("log |FFT|", fontsize=10)
    fig.tight_layout()
    fig.savefig("part2_2_frequency_analysis.png", dpi=100)

    # 3) Viewing distance: the same hybrid at 1, 1/2, 1/4, 1/8, 1/16 size
    panels([(shrunk(hyb, f), f"1/{f} size" if f > 1 else "full size") for f in (1, 2, 4, 8, 16)],
           (1, 5), (20, 5), "part2_2_viewing_distance.png")

    # 4) Colour: low only, high only, or both?
    variants = [("Gray + gray", g_low, g_high), ("Colour low + gray high", low_al, g_high),
                ("Gray low + colour high", g_low, high_al), ("Colour low + colour high", low_al, high_al)]
    results = [(t, hybrid(to3(lo), to3(hi), sl, sh)[2]) for t, lo, hi in variants]
    panels([(h, t) for t, h in results] + [(shrunk(h, 4), "same hybrid at 1/4 size") for _, h in results],
           (2, 4), (20, 11), "part2_2_color.png")

    cv2.imwrite("part2_2_hybrid.png", cv2.cvtColor((hyb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    plt.show()


if __name__ == "__main__":
    main()