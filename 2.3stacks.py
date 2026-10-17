"""Part 2.3: Gaussian and Laplacian stacks, and the blend that recreates Szeliski Fig. 3.42.

A STACK is a pyramid without downsampling: every level keeps the full image size.

    Gaussian stack   G_0 = I,   G_l = I filtered with a Gaussian of width sigma * 2**(l-1)
                     (Gaussians compose, so this equals filtering G_{l-1} again; the width
                     doubles per level like the blur of a downsampled pyramid)
    Laplacian stack  L_l = G_l - G_(l+1)   (one band of frequencies),   L_N = G_N (what is left)
                     so  sum_l L_l = I  exactly.

Blend (Burt & Adelson) with a mask M (1 where image A is wanted):
    blend = sum_l  L_l(A) * G_l(M)  +  L_l(B) * (1 - G_l(M))
The mask gets its own Gaussian stack, so the seam is soft at coarse levels and
sharp at fine ones.

Usage:  python part2_3.py IMAGE_A IMAGE_B [--sigma S] [--levels N] [--mask MASK.png]
  The default mask is a vertical step: IMAGE_A on the left half, IMAGE_B on the right.
  --mask is a grayscale image, white = take IMAGE_A, black = take IMAGE_B.
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


# ----------------------------------------------------------------------
# Stacks
# ----------------------------------------------------------------------
def gaussian_kernel(sigma):
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    return (g @ g.T).astype(np.float32)


def blur(img, sigma):
    """Gaussian blur with mirrored borders, for 2D or 3D (colour) arrays."""
    if img.ndim == 3:
        return np.dstack([blur(img[..., c], sigma) for c in range(img.shape[2])])
    k = gaussian_kernel(sigma)
    p = k.shape[0] // 2
    return fftconvolve(np.pad(img, p, mode="reflect"), k, mode="valid").astype(np.float32)


def gaussian_stack(img, levels, sigma):
    """[G_0 .. G_levels]: full-size images, blurrier every level."""
    return [img] + [blur(img, sigma * 2 ** (l - 1)) for l in range(1, levels + 1)]


def laplacian_stack(gstack):
    """[L_0 .. L_levels]: bands, plus the low-frequency remainder as the last entry."""
    return [gstack[l] - gstack[l + 1] for l in range(len(gstack) - 1)] + [gstack[-1]]


def blend_stacks(lap_a, lap_b, mask_stack):
    """Per-level blend; returns the weighted bands of A, of B, and their sum."""
    wa = [la * m for la, m in zip(lap_a, mask_stack)]
    wb = [lb * (1 - m) for lb, m in zip(lap_b, mask_stack)]
    return wa, wb, [a + b for a, b in zip(wa, wb)]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def clip(x):
    return np.clip(x, 0, 1)


def band(x, gain=3.0):
    """Show a signed band around mid-gray."""
    return clip(0.5 + gain * x)


def show(ax, img, title=None, frame=False):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=9)
    if frame:                       # keep a border so a white image is visible on a white page
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian width of the first level (doubles each level)")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--crop-bottom", type=int, default=0,
                        help="trim this many pixel rows from the bottom of both images "
                             "(e.g. to remove a scan caption or border)")
    args = parser.parse_args()

    a = load_rgb(args.image_a)
    b = load_rgb(args.image_b)
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if args.crop_bottom:
        a, b = a[:-args.crop_bottom], b[:-args.crop_bottom]
    h, w = a.shape[:2]

    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    else:
        mask = np.zeros((h, w), np.float32)
        mask[:, : w // 2] = 1.0
    mask3 = np.dstack([mask] * 3)

    N, s = args.levels, args.sigma
    ga, gb, gm = (gaussian_stack(x, N, s) for x in (a, b, mask3))
    la, lb = laplacian_stack(ga), laplacian_stack(gb)
    wa, wb, wsum = blend_stacks(la, lb, gm)
    blended = clip(sum(wsum))

    # --- sanity checks -------------------------------------------------------
    print(f"Image {w}x{h}, {N} levels, level sigmas: " +
          ", ".join(f"{s * 2 ** (l - 1):g}" for l in range(1, N + 1)))
    print(f"Laplacian stack sums back to the image: max error "
          f"{np.abs(sum(la) - a).max():.2e} (A), {np.abs(sum(lb) - b).max():.2e} (B)")
    ones = blend_stacks(la, lb, [np.ones_like(m) for m in gm])[2]
    print(f"Blend with an all-ones mask returns A: max error {np.abs(sum(ones) - a).max():.2e}")

    # --- Figure 1: the stacks ----------------------------------------------------
    cols = N + 1
    fig, axes = plt.subplots(5, cols, figsize=(2.6 * cols, 2.3 * 5 * h / w + 1))
    for row, (stack, title, f) in enumerate([
            (ga, "Gaussian stack A", clip), (la, "Laplacian stack A", band),
            (gb, "Gaussian stack B", clip), (lb, "Laplacian stack B", band),
            (gm, "Mask Gaussian stack", clip)]):
        kind = "G" if "Gaussian" in title else "L"
        name = "M" if "Mask" in title else title.split()[-1]
        for l in range(cols):
            residual = kind == "L" and l == N          # the last Laplacian entry is a plain image
            im = clip(stack[l]) if residual else f(stack[l])
            show(axes[row, l], im, f"{name}  {kind}{l}" + (" (residual)" if residual else ""),
                 frame=(name == "M"))
    fig.suptitle("Gaussian and Laplacian stacks (Laplacian bands shown around mid-gray, x3 contrast)")
    fig.tight_layout()
    fig.savefig("part2_3_stacks.png", dpi=110)

    # --- Figure 2: recreate Fig. 3.42 (a)-(l) --------------------------------------
    # rows 1-3: Laplacian levels 0, 2, 4 (high, medium, low); row 4: everything summed
    mid = [0, 2, min(4, N - 1)]
    letters = iter("abcdefghijkl")
    fig, axes = plt.subplots(4, 3, figsize=(10, 4 * 3.1 * h / w + 0.6))
    for r, l in enumerate(mid):
        for c, (stack, ttl) in enumerate([(wa, "A weighted by mask"),
                                          (wb, "B weighted by (1 - mask)"),
                                          (wsum, "sum")]):
            show(axes[r, c], band(stack[l]), f"({next(letters)})  level {l}  {ttl}")
    for c, (stack, ttl) in enumerate([(wa, "A, all levels"), (wb, "B, all levels"),
                                      (wsum, "blend")]):
        show(axes[3, c], clip(sum(stack)), f"({next(letters)})  {ttl}")
    fig.suptitle("Recreation of Szeliski Fig. 3.42: blending with Laplacian stacks")
    fig.tight_layout()
    fig.savefig("part2_3_figure_3_42.png", dpi=110)

    # --- Figure 3: inputs, mask, result -----------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    show(axes[0], a, "Image A")
    show(axes[1], b, "Image B")
    show(axes[2], mask, "Mask (white = image A)", frame=True)
    show(axes[3], blended, "Blended")
    fig.tight_layout()
    fig.savefig("part2_3_result.png", dpi=110)
    cv2.imwrite("part2_3_blend.png", cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show()


if __name__ == "__main__":
    main()