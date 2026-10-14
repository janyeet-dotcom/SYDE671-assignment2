"""Part 1.2: finite difference operators, gradient magnitude, and edge image.

Usage:
    python edges.py [path/to/cameraman.png] [--threshold T] [--sweep]
                    [--no-show] [--save-prefix PREFIX]

  --threshold T   binarization threshold on the gradient magnitude of the
                  image scaled to [0, 1] (default 0.25)
  --sweep         also show a grid of several thresholds to help choose one
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

DEFAULT_IMAGE = Path("Assignment_Images") / "cameraman.png"
DEFAULT_THRESHOLD = 0.25  # on gradient magnitude of an image scaled to [0, 1]

# Central-difference operators exactly as specified in the assignment.
DX = np.array([[1, 0, -1]], dtype=np.float64)
DY = np.array([[1], [0], [-1]], dtype=np.float64)


def load_grayscale(path: Path) -> np.ndarray:
    """Load as grayscale float in [0, 1]. Falls back to scikit-image's cameraman."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        try:
            from skimage import data
            print(f"{path} not found; using scikit-image's built-in cameraman instead.")
            gray = data.camera()
        except ImportError:
            raise FileNotFoundError(f"Could not read image: {path}")
    return gray.astype(np.float64) / 255.0


def gradients(img: np.ndarray):
    """Return (Ix, Iy, gradient magnitude) using zero-padded 'same' convolution."""
    ix = convolve2d(img, DX, mode="same", boundary="fill", fillvalue=0)
    iy = convolve2d(img, DY, mode="same", boundary="fill", fillvalue=0)
    return ix, iy, np.hypot(ix, iy)


def trim_border(arr: np.ndarray, width: int = 1) -> np.ndarray:
    """Zero-padding makes a fake strong edge along the image border; blank it out."""
    out = arr.copy()
    out[:width, :] = out[-width:, :] = 0
    out[:, :width] = out[:, -width:] = 0
    return out


def signed_to_display(arr: np.ndarray) -> np.ndarray:
    """Map a signed array to [0, 1] with 0 -> mid-gray (symmetric about zero)."""
    m = np.max(np.abs(arr))
    return 0.5 + 0.5 * arr / m if m > 0 else np.full_like(arr, 0.5)


def show(ax, img, title, vmin=0, vmax=1):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--sweep", action="store_true", help="show several thresholds")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-prefix", type=str, help="save figures as PREFIX_*.png")
    args = parser.parse_args()

    img = load_grayscale(args.image)
    ix, iy, mag = gradients(img)
    mag = trim_border(mag)
    edges = mag > args.threshold

    print(f"Image shape: {img.shape}, gradient magnitude range: "
          f"[{mag.min():.3f}, {mag.max():.3f}], threshold: {args.threshold}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    show(axes[0, 0], img, "Original")
    show(axes[0, 1], signed_to_display(ix), "Ix  (D_x)")
    show(axes[0, 2], signed_to_display(iy), "Iy  (D_y)")
    show(axes[1, 0], mag / mag.max(), "Gradient magnitude")
    show(axes[1, 1], edges, f"Edges (threshold = {args.threshold:g})")
    axes[1, 2].axis("off")
    fig.tight_layout()
    if args.save_prefix:
        fig.savefig(f"{args.save_prefix}_main.png", dpi=150)

    if args.sweep:
        thresholds = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
        for ax, t in zip(axes2.flat, thresholds):
            show(ax, mag > t, f"threshold = {t:g}")
        fig2.tight_layout()
        if args.save_prefix:
            fig2.savefig(f"{args.save_prefix}_sweep.png", dpi=150)

    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()