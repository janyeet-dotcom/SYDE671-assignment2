"""
Part 1.1: 2D convolution from scratch (four loops, two loops) vs. SciPy.

Usage: python 1.1convolution.py [image] [--full]
  --full  run the (very slow) four-loop version on the whole image
          (by default the comparison table uses a 64x64 crop)
"""
import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

BOX = np.ones((9, 9)) / 81.0
DX = np.array([[-1, 1]], dtype=float)
DY = np.array([[-1], [1]], dtype=float)


def pad_zeros(image, kshape):
    """Zero-pad so the output is 'same' size (extra row/col goes after, like SciPy)."""
    kh, kw = kshape
    return np.pad(image, ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2)))


def conv_four_loops(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    padded, flipped = pad_zeros(image, kernel.shape), kernel[::-1, ::-1]
    out = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            for i in range(kh):
                for j in range(kw):
                    out[r, c] += padded[r + i, c + j] * flipped[i, j]
    return out


def conv_two_loops(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    padded, flipped = pad_zeros(image, kernel.shape), kernel[::-1, ::-1]
    out = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            out[r, c] = np.sum(padded[r:r + kh, c:c + kw] * flipped)
    return out


def timed(f, *args):
    t = time.perf_counter()
    return f(*args), time.perf_counter() - t


def compare(gray, full):
    """Time all three implementations and check they agree with SciPy."""
    test = gray if full else gray[:64, :64]
    print(f"Comparison on a {test.shape[0]}x{test.shape[1]} image")
    print(f"{'filter':<8}{'4-loop (s)':>11}{'2-loop (s)':>11}{'SciPy (s)':>11}"
          f"{'max|4L-SciPy|':>15}{'max|2L-SciPy|':>15}")
    for name, k in {"Box 9x9": BOX, "Dx": DX, "Dy": DY}.items():
        o4, t4 = timed(conv_four_loops, test, k)
        o2, t2 = timed(conv_two_loops, test, k)
        ref, ts = timed(convolve2d, test, k, "same", "fill", 0)
        print(f"{name:<8}{t4:>11.3f}{t2:>11.3f}{ts:>11.4f}"
              f"{np.abs(o4 - ref).max():>15.1e}{np.abs(o2 - ref).max():>15.1e}")


def to_display(a):
    """Rescale any array to [0, 1] for imshow."""
    return (a - a.min()) / max(a.max() - a.min(), 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default="Assignment_Images/Janet.jpg")
    p.add_argument("--full", action="store_true")
    args = p.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(args.image)
    gray = gray.astype(float)
    compare(gray, args.full)

    # Full-image results use the two-loop version (identical output, much faster).
    blurred, ix, iy = (conv_two_loops(gray, k) for k in (BOX, DX, DY))
    panels = [("Original (grayscale)", gray), ("Box blur 9x9", blurred), ("Ix (Dx)", ix),
              ("Iy (Dy)", iy), ("Gradient magnitude", np.hypot(ix, iy))]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(to_display(img), cmap="gray")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("part1_1_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()