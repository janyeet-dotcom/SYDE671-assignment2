"""Part 2.2: hybrid images (Oliva, Torralba & Schyns, SIGGRAPH 2006).

    low  = G_low  * image_a                (Gaussian low-pass: what you see from far away)
    high = image_b - G_high * image_b      (impulse minus Gaussian: what you see up close)
    hybrid = low + high

Pipeline: align both photos on two reference points (the eyes) -> filter -> add.

Usage:
    python part2_2.py LOW_IMAGE HIGH_IMAGE [--sigma-low S] [--sigma-high S]
                      [--pts-low x1 y1 x2 y2] [--pts-high x1 y1 x2 y2]

  LOW_IMAGE   keeps only its coarse structure (visible from far away)
  HIGH_IMAGE  keeps only its fine detail (visible up close)
  --pts-*     the two alignment points (e.g. the eyes) in each image, in pixels.
              Leave them out to click the points yourself (needs a display).
  --no-align  use this for pairs that are already aligned (e.g. the standard
              cat/dog or Einstein/Marilyn files): no points, no warping.
Sigma is the Gaussian width in pixels of the aligned canvas (see CANVAS below);
larger sigma = lower cutoff frequency (cutoff is about 1 / (2 pi sigma) cycles/pixel).
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

# Both images are warped onto this canvas so the two reference points land on
# the same spot: same size, same orientation (horizontal), same position.
CANVAS_W, CANVAS_H = 520, 640
REF_CENTER = (260, 250)        # where the midpoint between the two points goes
REF_DIST = 150                 # distance in pixels between the two points


# ----------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------
def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def pick_points(img, title):
    """Click two reference points (e.g. the eyes) on an image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    pts = plt.ginput(2, timeout=0)
    plt.close()
    return np.array(pts, dtype=np.float32)


def align(img, pts):
    """Rotate, scale and translate img so pts land on the canonical positions.

    (This is align_images.m done as one similarity transform: the two points
    fix the rotation, scale and translation exactly.)
    """
    cx, cy = REF_CENTER
    target = np.float32([[cx - REF_DIST / 2, cy], [cx + REF_DIST / 2, cy]])
    M, _ = cv2.estimateAffinePartial2D(np.float32(pts), target, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------
def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return (g @ g.T).astype(np.float32)


def blur(img, sigma):
    """Gaussian low-pass (mirrored borders), for gray (2D) or colour (3D) images."""
    k = gaussian_kernel(sigma)
    p = k.shape[0] // 2
    if img.ndim == 2:
        return fftconvolve(np.pad(img, p, mode="reflect"), k, mode="valid")
    return np.dstack([blur(img[..., c], sigma) for c in range(img.shape[2])])


def hybrid(low_img, high_img, sigma_low, sigma_high):
    low = blur(low_img, sigma_low)
    high = high_img - blur(high_img, sigma_high)
    return low, high, np.clip(low + high, 0, 1)


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def log_spectrum(g):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)


def show(ax, img, title):
    ax.imshow(img if img.ndim == 3 else np.clip(img, 0, 1),
              cmap=None if img.ndim == 3 else "gray",
              vmin=None if img.ndim == 3 else 0, vmax=None if img.ndim == 3 else 1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def shrink_strip(img, factors=(1, 2, 4, 8, 16)):
    """The hybrid at several sizes side by side on white: a stand-in for stepping
    back from the screen (each image is the same picture, just smaller)."""
    h, w = img.shape[:2]
    gap = 20
    strip = np.ones((h, sum(w // f for f in factors) + gap * (len(factors) - 1)), np.float32)
    if img.ndim == 3:
        strip = np.dstack([strip] * 3)
    x = 0
    for f in factors:
        small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
        strip[h - small.shape[0]:, x:x + small.shape[1]] = small
        x += small.shape[1] + gap
    return strip


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("low_image")
    parser.add_argument("high_image")
    parser.add_argument("--sigma-low", type=float, default=7.0)
    parser.add_argument("--sigma-high", type=float, default=3.5)
    parser.add_argument("--pts-low", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--pts-high", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--no-align", action="store_true",
                        help="the pair is already aligned: skip the points and just "
                             "resize the high image to the size of the low image")
    args = parser.parse_args()

    low_rgb, high_rgb = load_rgb(args.low_image), load_rgb(args.high_image)

    # 1) Align ---------------------------------------------------------------
    if args.no_align:
        h, w = low_rgb.shape[:2]
        low_al = low_rgb
        high_al = (high_rgb if high_rgb.shape[:2] == (h, w)
                   else cv2.resize(high_rgb, (w, h), interpolation=cv2.INTER_AREA))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        show(axes[0], low_al, "Low-frequency image")
        show(axes[1], high_al, "High-frequency image")
    else:
        pts_low = (np.float32(args.pts_low).reshape(2, 2) if args.pts_low
                   else pick_points(low_rgb, "LOW-frequency image: click the two reference points"))
        pts_high = (np.float32(args.pts_high).reshape(2, 2) if args.pts_high
                    else pick_points(high_rgb, "HIGH-frequency image: click the same two points"))
        print("Reference points (reuse with --pts-low / --pts-high):")
        print("  --pts-low ", *pts_low.ravel().round(1))
        print("  --pts-high", *pts_high.ravel().round(1))

        low_al, high_al = align(low_rgb, pts_low), align(high_rgb, pts_high)

        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        for ax, im, p, t in ((axes[0], low_rgb, pts_low, "Low-frequency image"),
                             (axes[2], high_rgb, pts_high, "High-frequency image")):
            show(ax, im, t + " (original)")
            ax.plot(*p.T, "r+", ms=14, mew=2)
        show(axes[1], low_al, "aligned")
        show(axes[3], high_al, "aligned")
    fig.tight_layout()
    fig.savefig("part2_2_alignment.png", dpi=110)

    # 2) Hybrid (grayscale), with the frequency analysis ------------------------
    g_low, g_high = gray(low_al), gray(high_al)
    low, high, hyb = hybrid(g_low, g_high, args.sigma_low, args.sigma_high)

    cols = [(g_low, "Low image (aligned)"), (g_high, "High image (aligned)"),
            (low, f"Low-passed (sigma={args.sigma_low:g})"),
            (np.clip(0.5 + 2 * high, 0, 1), f"High-passed (sigma={args.sigma_high:g})"),
            (hyb, "Hybrid image")]
    fig, axes = plt.subplots(2, 5, figsize=(22, 10.5))
    for j, (im, t) in enumerate(cols):
        show(axes[0, j], im, t)
        spec = log_spectrum(im if j != 3 else high)
        axes[1, j].imshow(spec, cmap="gray")
        axes[1, j].set_title("log |FFT|", fontsize=10)
        axes[1, j].axis("off")
    fig.tight_layout()
    fig.savefig("part2_2_frequency_analysis.png", dpi=100)

    # 3) Viewing distance: the same hybrid at several sizes ------------------
    fig = plt.figure(figsize=(18, 5.5))
    plt.imshow(shrink_strip(hyb), cmap="gray", vmin=0, vmax=1)
    plt.title("Hybrid at 1x, 1/2, 1/4, 1/8, 1/16 size (like stepping back)")
    plt.axis("off")
    fig.tight_layout()
    fig.savefig("part2_2_viewing_distance.png", dpi=110)

    # 4) Colour: which component benefits from it? ---------------------------
    variants = [("Gray + gray", g_low, g_high),
                ("Colour low + gray high", low_al, g_high),
                ("Gray low + colour high", g_low, high_al),
                ("Colour low + colour high", low_al, high_al)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))
    for j, (t, lo_in, hi_in) in enumerate(variants):
        # a gray input is expanded to 3 channels so all variants can be added
        to3 = lambda a: a if a.ndim == 3 else np.dstack([a] * 3)
        _, _, h = hybrid(to3(lo_in), to3(hi_in), args.sigma_low, args.sigma_high)
        show(axes[0, j], h, t)
        # draw the 1/4-size copy at its true relative size inside a full-size frame
        hh, ww = h.shape[:2]
        small = cv2.resize(h, (ww // 4, hh // 4), interpolation=cv2.INTER_AREA)
        x0 = (ww - small.shape[1]) / 2
        axes[1, j].imshow(small, extent=(x0, x0 + small.shape[1], small.shape[0], 0))
        axes[1, j].set_xlim(0, ww)
        axes[1, j].set_ylim(hh, 0)
        axes[1, j].set_title("same hybrid at 1/4 size", fontsize=10)
        axes[1, j].axis("off")
    fig.tight_layout()
    fig.savefig("part2_2_color.png", dpi=100)

    cv2.imwrite("part2_2_hybrid.png", cv2.cvtColor((hyb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    plt.show()


if __name__ == "__main__":
    main()