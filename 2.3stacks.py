"""Parts 2.3 / 2.4: Gaussian and Laplacian stacks, and multiresolution blending.

A STACK is a pyramid without downsampling: every level keeps the full image size.

  Gaussian stack   G_0 = I,  G_l = I blurred with sigma * 2**(l-1)
  Laplacian stack  L_l = G_l - G_(l+1),  last entry L_N = G_N (low-frequency residual)
                   so sum_l L_l = I exactly.

Blend (Burt & Adelson) with mask M (1 where image A is wanted):
  blend = sum_l  L_l(A) * G_l(M)  +  L_l(B) * (1 - G_l(M))
The mask gets its own Gaussian stack, so the seam is soft at coarse levels and
sharp at fine ones.

Usage: python simple_2_3_stacks.py IMAGE_A IMAGE_B [--sigma S] [--levels N] [--mask MASK.png]
  Default mask: vertical step (A on the left half, B on the right).
  --mask: grayscale image, white = take A, black = take B (use for irregular masks).
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve


def blur(img, sigma):
    """Gaussian blur with mirrored borders (2D or colour)."""
    if img.ndim == 3:
        return np.dstack([blur(img[..., c], sigma) for c in range(img.shape[2])])
    ksize = 2 * int(np.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    p = ksize // 2
    return fftconvolve(np.pad(img, p, mode="reflect"), g @ g.T, mode="valid").astype(np.float32)


def gaussian_stack(img, levels, sigma):
    return [img] + [blur(img, sigma * 2 ** (l - 1)) for l in range(1, levels + 1)]


def laplacian_stack(g):
    return [g[l] - g[l + 1] for l in range(len(g) - 1)] + [g[-1]]


def blend_stacks(lap_a, lap_b, mask_stack):
    """Weighted bands of A, of B, and their per-level sum."""
    wa = [la * m for la, m in zip(lap_a, mask_stack)]
    wb = [lb * (1 - m) for lb, m in zip(lap_b, mask_stack)]
    return wa, wb, [a + b for a, b in zip(wa, wb)]


def load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def clip(x):
    return np.clip(x, 0, 1)


def band(x, gain=3.0):
    """Signed band shown around mid-gray."""
    return clip(0.5 + gain * x)


def draw(ax, img, title, frame=False):
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    if frame:   # keep a border so a white image is visible on a white page
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_a")
    p.add_argument("image_b")
    p.add_argument("--sigma", type=float, default=2.0, help="first-level sigma (doubles per level)")
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--mask")
    p.add_argument("--crop-bottom", type=int, default=0, help="trim rows (e.g. a scan caption)")
    args = p.parse_args()

    a, b = load_rgb(args.image_a), load_rgb(args.image_b)
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    if args.crop_bottom:
        a, b = a[:-args.crop_bottom], b[:-args.crop_bottom]
    h, w = a.shape[:2]

    if args.mask:
        m = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    else:
        mask = np.zeros((h, w), np.float32)
        mask[:, : w // 2] = 1
    N = args.levels

    ga, gb, gm = (gaussian_stack(x, N, args.sigma) for x in (a, b, np.dstack([mask] * 3)))
    la, lb = laplacian_stack(ga), laplacian_stack(gb)
    wa, wb, wsum = blend_stacks(la, lb, gm)
    blended = clip(sum(wsum))

    print(f"Image {w}x{h}, {N} levels, sigmas: "
          + ", ".join(f"{args.sigma * 2 ** (l - 1):g}" for l in range(1, N + 1)))
    print(f"Laplacian stack sums back to the image: max error "
          f"{np.abs(sum(la) - a).max():.1e} (A), {np.abs(sum(lb) - b).max():.1e} (B)")

    # Figure 1: all the stacks
    rows = [("Gaussian A", ga, clip), ("Laplacian A", la, band), ("Gaussian B", gb, clip),
            ("Laplacian B", lb, band), ("Mask", gm, clip)]
    fig, axes = plt.subplots(len(rows), N + 1, figsize=(2.6 * (N + 1), 2.4 * len(rows) * h / w + 1))
    for r, (name, stack, f) in enumerate(rows):
        for l in range(N + 1):
            residual = name.startswith("Laplacian") and l == N   # plain low-pass image
            im = clip(stack[l]) if residual else f(stack[l])
            draw(axes[r, l], im, f"{name} {l}" + (" (residual)" if residual else ""), frame=name == "Mask")
    fig.suptitle("Gaussian and Laplacian stacks (Laplacian bands at x3 contrast around mid-gray)")
    fig.tight_layout()
    fig.savefig("part2_3_stacks.png", dpi=110)

    # Figure 2: recreate Szeliski Fig. 3.42 (a)-(l)
    # rows 1-3: Laplacian levels 0, 2, 4 (high / medium / low); row 4: all levels summed
    letters = iter("abcdefghijkl")
    fig, axes = plt.subplots(4, 3, figsize=(10, 12.4 * h / w + 0.6))
    for r, l in enumerate([0, 2, min(4, N - 1)]):
        for c, (stack, t) in enumerate([(wa, "A x mask"), (wb, "B x (1 - mask)"), (wsum, "sum")]):
            draw(axes[r, c], band(stack[l]), f"({next(letters)}) level {l}: {t}")
    for c, (stack, t) in enumerate([(wa, "A, all levels"), (wb, "B, all levels"), (wsum, "blend")]):
        draw(axes[3, c], clip(sum(stack)), f"({next(letters)}) {t}")
    fig.suptitle("Recreation of Szeliski Fig. 3.42")
    fig.tight_layout()
    fig.savefig("part2_3_figure_3_42.png", dpi=110)

    # Figure 3: inputs, mask, result
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, im, t in zip(axes, (a, b, mask, blended),
                         ("Image A", "Image B", "Mask (white = A)", "Blended")):
        draw(ax, im, t, frame=t.startswith("Mask"))
    fig.tight_layout()
    fig.savefig("part2_3_result.png", dpi=110)
    cv2.imwrite("part2_3_blend.png", cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    plt.show()


if __name__ == "__main__":
    main()