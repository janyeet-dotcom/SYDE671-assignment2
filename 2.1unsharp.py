"""
Part 2.1: unsharp mask sharpening.

    blurred = I * G                 (Gaussian low-pass: low frequencies)
    detail  = I - I * G             (what the blur removed: high frequencies)
    sharp   = I + alpha * detail
            = I * ((1 + alpha) * delta - alpha * G)      <- one convolution

Usage: python 2.1unsharp.py IMAGE [--sigma S]
  Sigma is in pixels; default is (longest side) / 150.
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

ALPHA = 1.5                    # main result
ALPHAS = [0.5, 1, 2, 4, 8]     # comparison


def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return g @ g.T


def unsharp_kernel(sigma, alpha):
    """(1 + alpha) * delta - alpha * G as a single kernel."""
    G = gaussian_kernel(sigma)
    delta = np.zeros_like(G)
    delta[G.shape[0] // 2, G.shape[1] // 2] = 1
    return (1 + alpha) * delta - alpha * G


def convolve(img, kernel):
    """Same-size convolution per colour channel (mirrored borders, FFT for speed)."""
    p = kernel.shape[0] // 2
    padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.dstack([fftconvolve(padded[..., c], kernel, mode="valid") for c in range(3)])
    return np.clip(out, 0, 1)


def row(images, titles, name, size):
    fig, axes = plt.subplots(*size[0], figsize=size[1])
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, im, t in zip(np.ravel(axes), images, titles):
        ax.imshow(im)
        ax.set_title(t)
    fig.tight_layout()
    fig.savefig(name, dpi=120)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--sigma", type=float)
    args = p.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    sigma = args.sigma or round(max(img.shape[:2]) / 150, 1)
    stem = Path(args.image).stem
    print(f"Image {img.shape[1]}x{img.shape[0]}, sigma = {sigma:g}")

    blurred = convolve(img, gaussian_kernel(sigma))
    detail = img - blurred
    sharp = convolve(img, unsharp_kernel(sigma, ALPHA))

    row([img, blurred, np.clip(0.5 + 2 * detail, 0, 1), sharp],
        ["Original", f"Blurred (sigma={sigma:g})", "High frequencies (I - blurred)", f"Sharpened (alpha={ALPHA:g})"],
        f"part2_1_process_{stem}.png", ((1, 4), (18, 5)))

    row([img] + [convolve(img, unsharp_kernel(sigma, a)) for a in ALPHAS],
        ["alpha = 0 (original)"] + [f"alpha = {a:g}" for a in ALPHAS],
        f"part2_1_alphas_{stem}.png", ((2, 3), (15, 10)))

    # Evaluation: blur a sharp image, then try to sharpen it again
    resharp = convolve(blurred, unsharp_kernel(sigma, ALPHA))
    row([img, blurred, resharp], ["Original (sharp)", "Blurred", "Blurred, then sharpened"],
        f"part2_1_evaluation_{stem}.png", ((1, 3), (16, 5.5)))
    mse = lambda a: np.mean((a - img) ** 2)
    print(f"MSE vs. original: blurred {mse(blurred):.5f}, re-sharpened {mse(resharp):.5f}")
    plt.show()


if __name__ == "__main__":
    main()