"""
Part 1.3 (Derivative of Gaussian) and Bells & Whistles (gradient orientation).

1. G = 2D Gaussian (cv2.getGaussianKernel -> outer product).
2. Blur with G, then differentiate with D_x / D_y   (two convolutions).
3. DoG_x = G * D_x, DoG_y = G * D_y, applied in ONE convolution; verify it matches 2.
4. Compare gradient magnitude / edges: finite difference vs. DoG.
5. Orientation -> HSV hue without arctan/arctan2/np.angle (see pseudo_angle).

Usage: python 1.3gaussian.py [cameraman.png] [--sigma S] [--thr-fd T] [--thr-dog T]
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

DX = np.array([[1, 0, -1]], dtype=float)
DY = np.array([[1], [0], [-1]], dtype=float)


def conv(img, k):
    """Zero-padded 'same' convolution (same boundary handling as 1.1 / 1.2)."""
    return convolve2d(img, k, mode="same", boundary="fill", fillvalue=0)


def gaussian_2d(ksize, sigma):
    g = cv2.getGaussianKernel(ksize, sigma).ravel()
    return np.outer(g, g)


def trim(a, w):
    """Zero a border band where zero padding creates fake edges."""
    out = a.copy()
    out[:w] = out[-w:] = 0
    out[:, :w] = out[:, -w:] = 0
    return out


def signed(a):
    return 0.5 + 0.5 * a / np.abs(a).max()


def pseudo_angle(x, y):
    """Direction of the vector (x, y) as a number in [0, 1), using only arithmetic.

    "Diamond angle": p = x / (|x| + |y|) lies in [-1, 1] and rises monotonically
    with the true angle:  y >= 0 -> 1 - p  (0..2),   y < 0 -> 3 + p  (2..4).
    Dividing by 4 gives a hue. It is exact at 0/45/90/... degrees and off by at
    most a few degrees in between, which is fine for colouring.
    """
    l1 = np.abs(x) + np.abs(y)
    p = np.divide(x, l1, out=np.zeros_like(x, dtype=float), where=l1 > 0)
    return np.where(y >= 0, 1 - p, 3 + p) / 4


def orientation_rgb(ix, iy, mag):
    """HSV image: hue = gradient direction, saturation = 1, value = edge strength."""
    hue = pseudo_angle(ix, iy)
    value = np.clip(mag / np.percentile(mag, 99.5), 0, 1)
    return hsv_to_rgb(np.dstack([hue, np.ones_like(hue), value]))


def color_wheel(n=256):
    """Legend for the hue mapping."""
    ys, xs = np.mgrid[1:-1:n * 1j, -1:1:n * 1j]  # y points up
    hue = pseudo_angle(xs, ys)
    rgb = hsv_to_rgb(np.dstack([hue, np.ones_like(hue), np.ones_like(hue)]))
    rgb[np.hypot(xs, ys) > 1] = 1
    return rgb


def grid(images, titles, shape, size):
    fig, axes = plt.subplots(*shape, figsize=size)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, im, t in zip(np.ravel(axes), images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default="Assignment_Images/cameraman.png")
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--thr-fd", type=float, default=0.25, help="edge threshold, finite difference")
    p.add_argument("--thr-dog", type=float, default=0.12, help="edge threshold, Gaussian/DoG")
    args = p.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.image)
    img = img.astype(float) / 255
    sigma = args.sigma
    ksize = int(6 * sigma) // 2 * 2 + 1  # odd, about 6 sigma
    print(f"Gaussian: ksize={ksize}, sigma={sigma}")

    # Finite difference baseline
    fd_x, fd_y = conv(img, DX), conv(img, DY)
    fd_mag = trim(np.hypot(fd_x, fd_y), 1)

    # Blur then differentiate (two convolutions)
    G = gaussian_2d(ksize, sigma)
    blurred = conv(img, G)
    bl_x, bl_y = conv(blurred, DX), conv(blurred, DY)

    # DoG filters: one convolution each
    DoG_x, DoG_y = convolve2d(G, DX, mode="full"), convolve2d(G, DY, mode="full")
    dog_x, dog_y = conv(img, DoG_x), conv(img, DoG_y)

    # Verification. Convolution is associative, so (I*G)*D == I*(G*D); with zero
    # padding the two routes only differ at the border (padded once vs. twice).
    m = ksize // 2 + 2
    inner = (slice(m, -m), slice(m, -m))
    for n, a, b in (("x", bl_x, dog_x), ("y", bl_y, dog_y)):
        print(f"d{n}: max|blur-then-diff - DoG|  interior {np.abs(a - b)[inner].max():.2e}, "
              f"whole image {np.abs(a - b).max():.2e}")

    dog_mag = trim(np.hypot(dog_x, dog_y), m)
    fd_edges, dog_edges = fd_mag > args.thr_fd, dog_mag > args.thr_dog

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, k, t in zip(axes, (G, DoG_x, DoG_y),
                        (f"Gaussian G ({ksize}x{ksize}, sigma={sigma:g})", "DoG_x = G * D_x", "DoG_y = G * D_y")):
        ax.imshow(k, cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("part1_3_kernels.png", dpi=130)

    diff_x, diff_y = np.abs(bl_x - dog_x), np.abs(bl_y - dog_y)
    grid([signed(bl_x), signed(dog_x), diff_x / diff_x.max(),
          signed(bl_y), signed(dog_y), diff_y / diff_y.max()],
         ["Ix: blur, then D_x", "Ix: single DoG_x", "|difference|",
          "Iy: blur, then D_y", "Iy: single DoG_y", "|difference|"],
         (2, 3), (14, 9)).savefig("part1_3_dog_verification.png", dpi=130)

    grid([img, signed(fd_x), fd_mag / fd_mag.max(), fd_edges,
          blurred, signed(dog_x), dog_mag / dog_mag.max(), dog_edges],
         ["Original", "Ix (finite difference)", "Magnitude (FD)", f"Edges, FD (thr {args.thr_fd:g})",
          f"Gaussian-blurred (sigma={sigma:g})", "Ix (DoG_x)", "Magnitude (DoG)", f"Edges, DoG (thr {args.thr_dog:g})"],
         (2, 4), (18, 9)).savefig("part1_3_fd_vs_dog.png", dpi=130)

    # Orientation. Convolution with [1, 0, -1] gives I[x+1] - I[x-1] (x to the right).
    # Row index grows DOWNWARD but the colour wheel has y UP, so use -Iy:
    # (Ix, -Iy) is the true gradient, pointing from dark toward bright.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1, 0.7]})
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(orientation_rgb(dog_x, -dog_y, dog_mag))
    axes[1].set_title("Orientation (hue) x strength (brightness)")
    axes[2].imshow(color_wheel(), extent=(-1, 1, -1, 1))
    axes[2].set_title("Direction -> hue")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("part1_3_orientation.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()