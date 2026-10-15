"""Part 2.1: image sharpening with the unsharp mask filter.

    blurred = I * G                  (Gaussian low-pass: keeps low frequencies)
    detail  = I - I * G              (what the blur removed: high frequencies)
    sharp   = I + alpha * detail
            = I * ((1 + alpha) * delta - alpha * G)    <- one convolution

Usage:  python part2_1.py IMAGE [--sigma S]

Sigma is measured in pixels, so what looks like a "blur" depends on image size.
If --sigma is not given it is set to (longest side) / 150, which gives sigma = 2
for a 300 px image and about 17 for a 2500 px photo.
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

ALPHA = 1.5                           # sharpening amount for the main result
ALPHAS = [0.5, 1, 2, 4, 8]            # amounts shown in the comparison


def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1          # covers +/- 3 sigma
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T                                    # 2D = outer product


def unsharp_kernel(sigma, alpha):
    """(1 + alpha) * delta - alpha * G: the whole filter as a single kernel."""
    G = gaussian_kernel(sigma)
    delta = np.zeros_like(G)
    delta[G.shape[0] // 2, G.shape[1] // 2] = 1
    return (1 + alpha) * delta - alpha * G


def convolve(img, kernel):
    """Same-size convolution on each colour channel (mirrored borders).

    Uses the FFT method so large kernels on large photos stay fast; the result
    is the same convolution as convolve2d.
    """
    p = kernel.shape[0] // 2
    padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
    channels = [fftconvolve(padded[..., c], kernel, mode="valid")
                for c in range(img.shape[2])]
    return np.clip(np.dstack(channels), 0, 1)


def show(ax, img, title):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--sigma", type=float, default=None,
                        help="blur width in pixels (default: longest side / 150)")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    if args.sigma is None:
        args.sigma = round(max(img.shape[:2]) / 150, 1)
    print(f"Image {img.shape[1]}x{img.shape[0]}, sigma = {args.sigma:g}")
    G = gaussian_kernel(args.sigma)

    # 1) Blurred, high frequencies, sharpened ------------------------------
    blurred = convolve(img, G)
    detail = img - blurred
    sharp = convolve(img, unsharp_kernel(args.sigma, ALPHA))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    show(axes[0], img, "Original")
    show(axes[1], blurred, f"Blurred (sigma={args.sigma:g})")
    show(axes[2], np.clip(0.5 + 2 * detail, 0, 1), "High frequencies (I - blurred)")
    show(axes[3], sharp, f"Sharpened (alpha={ALPHA:g})")
    fig.tight_layout()
    fig.savefig("part2_1_process.png", dpi=120)

    # 2) Effect of the sharpening amount -----------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    show(axes[0, 0], img, "alpha = 0 (original)")
    for ax, a in zip(axes.flat[1:], ALPHAS):
        show(ax, convolve(img, unsharp_kernel(args.sigma, a)), f"alpha = {a:g}")
    fig.tight_layout()
    fig.savefig("part2_1_alphas.png", dpi=120)

    # 3) Evaluation: blur the sharp image, then sharpen it again -----------
    resharpened = convolve(blurred, unsharp_kernel(args.sigma, ALPHA))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    show(axes[0], img, "Original (sharp)")
    show(axes[1], blurred, "Blurred")
    show(axes[2], resharpened, "Blurred, then sharpened")
    fig.tight_layout()
    fig.savefig("part2_1_evaluation.png", dpi=120)

    mse = lambda a: np.mean((a - img) ** 2)
    print(f"MSE vs. original: blurred {mse(blurred):.5f}, "
          f"re-sharpened {mse(resharpened):.5f}")
    plt.show()


if __name__ == "__main__":
    main()