"""Part 1.3 (Derivative of Gaussian) and Bells & Whistles (gradient orientation).

Pipeline
    1. Build a 2D Gaussian G (cv2.getGaussianKernel -> outer product).
    2. Blur the image with G, then differentiate with D_x / D_y (two convolutions).
    3. Build DoG filters  DoG_x = G * D_x,  DoG_y = G * D_y  and apply them in a
       single convolution. Verify the result matches step 2.
    4. Compare gradient magnitude / edge maps: finite difference vs. DoG.
    5. Visualise gradient orientation with HSV (no built-in angle functions).

Usage:
    python part1_3.py [cameraman.png] [--sigma S] [--ksize K]
                      [--threshold-fd T] [--threshold-dog T] [--sweep]
                      [--no-show] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d

DEFAULT_IMAGE = Path("Assignment_Images") / "cameraman.png"

# Finite-difference operators exactly as given in the assignment.
DX = np.array([[1, 0, -1]], dtype=np.float64)
DY = np.array([[1], [0], [-1]], dtype=np.float64)


# ----------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------
def load_grayscale(path: Path) -> np.ndarray:
    """Load as grayscale float in [0, 1]."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return gray.astype(np.float64) / 255.0


def conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-padded 'same' convolution (same boundary handling as Part 1.1/1.2)."""
    return convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)


def gaussian_kernel_2d(ksize: int, sigma: float) -> np.ndarray:
    """2D Gaussian = outer product of a 1D Gaussian with itself (sums to 1)."""
    g1d = cv2.getGaussianKernel(ksize, sigma).ravel()
    return np.outer(g1d, g1d)


def trim_border(arr: np.ndarray, width: int) -> np.ndarray:
    """Zero out a border band where zero padding creates fake edges."""
    out = arr.copy()
    out[:width, :] = out[-width:, :] = 0
    out[:, :width] = out[:, -width:] = 0
    return out


def signed_to_display(arr: np.ndarray) -> np.ndarray:
    """Map a signed array to [0, 1], with 0 -> mid-gray."""
    m = np.max(np.abs(arr))
    return 0.5 + 0.5 * arr / m if m > 0 else np.full_like(arr, 0.5)


def show(ax, img, title, vmin=0.0, vmax=1.0):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ----------------------------------------------------------------------
# Bells & whistles: gradient orientation without built-in angle functions
# ----------------------------------------------------------------------
def pseudo_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map the direction of the vector (x, y) to [0, 1) using only arithmetic.

    Uses the "diamond angle": p = x / (|x| + |y|) lies in [-1, 1] and varies
    monotonically with the true angle, so
        y >= 0 :  4 * hue = 1 - p     (angles 0 .. 180 deg -> 0 .. 2)
        y <  0 :  4 * hue = 3 + p     (angles 180 .. 360 deg -> 2 .. 4)
    No arctan / arctan2 / np.angle is involved. It is monotonic and hits the
    right values at 0/45/90/... degrees, but is not perfectly linear in the
    true angle (at most ~4 degrees of distortion), which is fine for colouring.
    """
    l1 = np.abs(x) + np.abs(y)
    p = np.divide(x, l1, out=np.zeros_like(x, dtype=np.float64), where=l1 > 0)
    return np.where(y >= 0, 1.0 - p, 3.0 + p) / 4.0


def orientation_to_rgb(ix: np.ndarray, iy: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """HSV image: hue = gradient direction, saturation = 1, value = edge strength."""
    hue = pseudo_angle(ix, iy)
    value = np.clip(mag / np.percentile(mag, 99.5), 0.0, 1.0)
    hsv = np.dstack([hue, np.ones_like(hue), value])
    return hsv_to_rgb(hsv)


def color_wheel(size: int = 256) -> np.ndarray:
    """Legend: the same hue mapping drawn over a disc of directions."""
    ys, xs = np.mgrid[1:-1:size * 1j, -1:1:size * 1j]  # y up
    radius = np.hypot(xs, ys)
    hue = pseudo_angle(xs, ys)
    rgb = hsv_to_rgb(np.dstack([hue, np.ones_like(hue), np.ones_like(hue)]))
    rgb[radius > 1] = 1.0  # white outside the disc
    return rgb


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--ksize", type=int, default=None,
                        help="odd Gaussian kernel size (default: about 6*sigma + 1)")
    parser.add_argument("--threshold-fd", type=float, default=0.25,
                        help="edge threshold for plain finite differences")
    parser.add_argument("--threshold-dog", type=float, default=0.12,
                        help="edge threshold for the Gaussian / DoG gradients")
    parser.add_argument("--sweep", action="store_true",
                        help="also save a grid of DoG thresholds")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="save figures here as part1_3_*.png")
    args = parser.parse_args()

    ksize = args.ksize or int(6 * args.sigma) // 2 * 2 + 1  # odd
    img = load_grayscale(args.image)
    print(f"Image {img.shape}, Gaussian ksize={ksize}, sigma={args.sigma}")

    # --- Part 1.2 baseline: plain finite differences ---------------------
    fd_x, fd_y = conv(img, DX), conv(img, DY)
    fd_mag = trim_border(np.hypot(fd_x, fd_y), 1)

    # --- Part 1.3: Gaussian, then differentiate (two convolutions) -------
    G = gaussian_kernel_2d(ksize, args.sigma)
    blurred = conv(img, G)
    bl_x, bl_y = conv(blurred, DX), conv(blurred, DY)

    # --- DoG filters: G * D_x, G * D_y (full convolution keeps every tap) -
    DoG_x = convolve2d(G, DX, mode="full")
    DoG_y = convolve2d(G, DY, mode="full")
    dog_x, dog_y = conv(img, DoG_x), conv(img, DoG_y)

    # --- Verification ----------------------------------------------------
    # Convolution is associative, so (I*G)*D == I*(G*D). With zero padding this
    # is exact away from the border; right at the border the two routes pad
    # differently (blur pads once then differentiate pads again, DoG pads once).
    m = ksize // 2 + 2
    inner = (slice(m, -m), slice(m, -m))
    for name, a, b in (("x", bl_x, dog_x), ("y", bl_y, dog_y)):
        print(f"d{name}: max |blur-then-diff - DoG|  interior = "
              f"{np.max(np.abs(a[inner] - b[inner])):.2e},  "
              f"whole image = {np.max(np.abs(a - b)):.2e}  "
              f"(interior match: {np.allclose(a[inner], b[inner], atol=1e-10)})")

    margin = ksize // 2 + 2
    dog_mag = trim_border(np.hypot(dog_x, dog_y), margin)
    bl_mag = trim_border(np.hypot(bl_x, bl_y), margin)
    fd_edges = fd_mag > args.threshold_fd
    dog_edges = dog_mag > args.threshold_dog
    print(f"Magnitude max: finite diff {fd_mag.max():.3f}, DoG {dog_mag.max():.3f}")

    # --- Figures ---------------------------------------------------------
    figs = {}

    # 1) The filters themselves
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, k, t in zip(axes, (G, DoG_x, DoG_y),
                        (f"Gaussian G ({ksize}x{ksize}, sigma={args.sigma:g})",
                         "DoG_x = G * D_x", "DoG_y = G * D_y")):
        ax.imshow(k, cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    figs["kernels"] = fig

    # 2) Smoothing first vs. single DoG convolution
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    pairs = (("x", bl_x, dog_x), ("y", bl_y, dog_y))
    for row, (n, a, b) in zip(axes, pairs):
        diff = np.abs(a - b)
        show(row[0], signed_to_display(a), f"I{n}: blur, then D_{n}")
        show(row[1], signed_to_display(b), f"I{n}: single DoG_{n} convolution")
        show(row[2], diff / max(diff.max(), 1e-12),
             f"|difference| (interior max {np.max(diff[inner]):.1e})")
    fig.tight_layout()
    figs["dog_verification"] = fig

    # 3) Finite difference vs. Gaussian-smoothed / DoG
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    show(axes[0, 0], img, "Original")
    show(axes[0, 1], signed_to_display(fd_x), "Ix  (finite difference)")
    show(axes[0, 2], fd_mag / fd_mag.max(), "Gradient magnitude (FD)")
    show(axes[0, 3], fd_edges, f"Edges, FD (threshold {args.threshold_fd:g})")
    show(axes[1, 0], blurred, f"Gaussian-blurred (sigma={args.sigma:g})")
    show(axes[1, 1], signed_to_display(dog_x), "Ix  (DoG_x)")
    show(axes[1, 2], dog_mag / dog_mag.max(), "Gradient magnitude (DoG)")
    show(axes[1, 3], dog_edges, f"Edges, DoG (threshold {args.threshold_dog:g})")
    fig.tight_layout()
    figs["fd_vs_dog"] = fig

    # 4) Gradient orientation (bells & whistles), from the smoothed gradient
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [1, 1, 0.7]})
    show(axes[0], img, "Original")
    # Sign conventions: true convolution with [1, 0, -1] computes
    # I[x+1] - I[x-1], the ordinary derivative with x to the right. Along y the
    # row index grows DOWNWARD, but the colour wheel is drawn with y pointing
    # UP, so we negate Iy. (Ix, -Iy) is then the true gradient, which points
    # from dark toward bright.
    axes[1].imshow(orientation_to_rgb(dog_x, -dog_y, dog_mag))
    axes[1].set_title("Gradient orientation (hue) x strength (brightness)", fontsize=10)
    axes[1].axis("off")
    axes[2].imshow(color_wheel(), extent=(-1, 1, -1, 1))
    axes[2].set_title("Direction -> hue", fontsize=10)
    axes[2].axis("off")
    fig.tight_layout()
    figs["orientation"] = fig

    # 5) Optional threshold sweep for the DoG magnitude
    if args.sweep:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        top = dog_mag.max()
        for ax, f in zip(axes.flat, (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)):
            t = f * top / 0.5  # thresholds scale with the peak of the magnitude
            show(ax, dog_mag > t, f"threshold = {t:.3f}")
        fig.tight_layout()
        figs["dog_sweep"] = fig

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, f in figs.items():
            f.savefig(args.out_dir / f"part1_3_{name}.png", dpi=130)
        print(f"Saved {len(figs)} figures to {args.out_dir}")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()