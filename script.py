"""2D convolution from scratch: four loops, two loops, and a SciPy comparison.

Recap: for an image f and kernel g, the 2D convolution is
    (f * g)[r, c] = sum_i sum_j f[r - i, c - j] * g[i, j]
i.e. flip the kernel in both directions, slide it over the image, and at each
position sum the elementwise products. Borders are handled by zero padding so
the output has the same size as the input ("same").

Usage:
    python script.py [path/to/image.jpg] [--no-show] [--save OUTPUT.png]
                     [--full-four-loops]
"""

import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

DEFAULT_IMAGE = Path("Assignment_Images") / "Janet.jpg"


# ----------------------------------------------------------------------
# Convolution implementations
# ----------------------------------------------------------------------
def pad_zeros(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Zero-pad so a 'same'-size output can be produced.

    Even kernel sizes put the extra row/column after the image, which matches
    scipy.signal.convolve2d(..., mode="same").
    """
    kh, kw = kernel_shape
    pad = ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2))
    return np.pad(image, pad, mode="constant", constant_values=0)


def convolve2d_four_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution with four explicit for loops (image rows/cols, kernel rows/cols)."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    h, w = image.shape
    kh, kw = kernel.shape

    padded = pad_zeros(image, kernel.shape)
    flipped = kernel[::-1, ::-1]  # flip both axes: true convolution

    output = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            total = 0.0
            for i in range(kh):
                for j in range(kw):
                    total += padded[r + i, c + j] * flipped[i, j]
            output[r, c] = total
    return output


def convolve2d_two_loops(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolution with two explicit for loops; the kernel sum is done with NumPy."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    h, w = image.shape
    kh, kw = kernel.shape

    padded = pad_zeros(image, kernel.shape)
    flipped = kernel[::-1, ::-1]

    output = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            output[r, c] = np.sum(padded[r:r + kh, c:c + kw] * flipped)
    return output


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Linearly rescale an array to uint8 [0, 255] for visualization."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.round(255 * (img - lo) / (hi - lo)).astype(np.uint8)


def load_grayscale(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return gray.astype(np.float64)


def timed(func, *args):
    start = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - start


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------
BOX_9x9 = np.ones((9, 9), dtype=np.float64) / 81.0
DX = np.array([[-1, 1]], dtype=np.float64)
DY = np.array([[-1], [1]], dtype=np.float64)

FILTERS = {"Box 9x9": BOX_9x9, "Dx": DX, "Dy": DY}


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def compare_implementations(gray: np.ndarray, full_four_loops: bool) -> None:
    """Check four-loop, two-loop and SciPy results against each other.

    The four-loop version is very slow in pure Python, so by default it is
    checked on a small crop; pass --full-four-loops to use the whole image.
    """
    test = gray if full_four_loops else gray[:64, :64]
    print(f"\nComparison on {test.shape[0]}x{test.shape[1]} image "
          f"({'full' if full_four_loops else 'crop'})")
    print(f"{'Filter':<9}{'4-loop (s)':>11}{'2-loop (s)':>11}"
          f"{'SciPy (s)':>11}  {'4L=SciPy':<9}{'2L=SciPy':<9}{'max diff (2L)'}")

    for name, kernel in FILTERS.items():
        out4, t4 = timed(convolve2d_four_loops, test, kernel)
        out2, t2 = timed(convolve2d_two_loops, test, kernel)
        ref, ts = timed(convolve2d, test, kernel, "same", "fill", 0)

        ok4 = np.allclose(out4, ref, atol=1e-10)
        ok2 = np.allclose(out2, ref, atol=1e-10)
        diff = np.max(np.abs(out2 - ref))
        print(f"{name:<9}{t4:>11.3f}{t2:>11.3f}{ts:>11.4f}  "
              f"{str(ok4):<9}{str(ok2):<9}{diff:.2e}")


def show_results(gray, blurred, ix, iy, grad_mag, save_path=None, show=True):
    panels = [
        ("Original (grayscale)", gray),
        ("Box blur 9x9", blurred),
        ("Ix (Dx)", ix),
        ("Iy (Dy)", iy),
        ("Gradient magnitude", grad_mag),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(normalize_for_display(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    axes.flat[-1].axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--no-show", action="store_true", help="don't open a plot window")
    parser.add_argument("--save", type=Path, help="save the result figure to this path")
    parser.add_argument("--full-four-loops", action="store_true",
                        help="run the four-loop comparison on the whole image (slow)")
    args = parser.parse_args()

    print(f"Loading {args.image} ...")
    gray = load_grayscale(args.image)
    print("Image shape:", gray.shape)

    compare_implementations(gray, args.full_four_loops)

    # Full-image results use the two-loop version (same output as four loops).
    blurred = convolve2d_two_loops(gray, BOX_9x9)
    ix = convolve2d_two_loops(gray, DX)
    iy = convolve2d_two_loops(gray, DY)
    grad_mag = np.hypot(ix, iy)

    show_results(gray, blurred, ix, iy, grad_mag, args.save, not args.no_show)


if __name__ == "__main__":
    main()