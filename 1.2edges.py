"""
Part 1.2: finite difference operators -> Ix, Iy, gradient magnitude, edge image.

Usage: python 1.2edges.py [cameraman.png] [--threshold T]
  The threshold applies to the gradient magnitude of the image scaled to [0, 1].
  A grid of several thresholds is always saved too, to help choose one.
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

DX = np.array([[1, 0, -1]], dtype=float)
DY = np.array([[1], [0], [-1]], dtype=float)


def signed_to_display(a):
    """Signed array -> [0, 1] with 0 at mid-gray."""
    return 0.5 + 0.5 * a / np.abs(a).max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default="Assignment_Images/cameraman.png")
    p.add_argument("--threshold", type=float, default=0.3)
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)
    img = img.astype(float) / 255

    ix = convolve2d(img, DX, mode="same", boundary="fill", fillvalue=0)
    iy = convolve2d(img, DY, mode="same", boundary="fill", fillvalue=0)
    mag = np.hypot(ix, iy)
    # Zero padding creates a fake strong edge on the image border: blank it.
    mag[[0, -1], :] = 0
    mag[:, [0, -1]] = 0
    edges = mag > args.threshold

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = [("Original", img), ("Ix (D_x)", signed_to_display(ix)),
              ("Iy (D_y)", signed_to_display(iy)), ("Gradient magnitude", mag / mag.max()),
              (f"Edges (threshold = {args.threshold:g})", edges)]
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, im) in zip(axes.flat, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("part1_2_main.png", dpi=150)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, t in zip(axes.flat, [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]):
        ax.imshow(mag > t, cmap="gray")
        ax.set_title(f"threshold = {t:g}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("part1_2_sweep.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()