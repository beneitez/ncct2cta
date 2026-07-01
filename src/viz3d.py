"""3D-style visualizations of the CT data (vessels are the natural 3D structure in CTA).

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

For a subject (brain-masked so the skull doesn't dominate the projections):
  * orthogonal MIP triptych (axial / coronal / sagittal), anatomically proportioned,
  * a rotating maximum-intensity-projection GIF (spinning vessel tree about the S-I axis),
  * a static montage of rotation angles (so the rotation is viewable as a single image).

With --pred <dir> it renders the ground-truth CTA and the model's predicted CTA SIDE BY SIDE
(GT top/left, prediction bottom/right) for a direct visual comparison.

Run:  python -m src.viz3d --sid 0001 --pred preds --cmap viridis --out artifacts/figures3d
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from . import constants as C
from .brain_mask import get_brain_mask
from .data import discover_subjects, load_nii

_VESSEL_WIN = (0, 400)  # HU window that shows iodine contrast
_BG = -1000.0           # value for non-brain / rotation fill (never wins a max-projection)


def _norm(hu, win=_VESSEL_WIN):
    return np.clip((hu - win[0]) / (win[1] - win[0]), 0.0, 1.0)


def _colorize(mip01: np.ndarray, cmap: str) -> np.ndarray:
    """Map a normalized MIP to RGB uint8, forcing background to black (so viridis' dark
    purple low end doesn't fill the background)."""
    rgb = (matplotlib.colormaps[cmap](np.clip(mip01, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[mip01 < 0.01] = 0
    return rgb


def _ortho_views(vol_hu, brain, a):
    """Return [(mip01, title, aspect)] for axial/coronal/sagittal. axes 0=L-R,1=A-P,2=S-I."""
    v = np.where(brain > 0, vol_hu, _BG)
    return [
        (_norm(v.max(axis=2)).T, "Axial MIP (top-down)", 1.0),   # (A-P, L-R)
        (_norm(v.max(axis=1)).T, "Coronal MIP (front)", a),      # (S-I, L-R)
        (_norm(v.max(axis=0)).T, "Sagittal MIP (side)", a),      # (S-I, A-P)
    ]


def ortho_mip(vol_hu, brain, zooms, out_path, cmap="viridis"):
    a = float(zooms[2]) / float(zooms[0])
    views = _ortho_views(vol_hu, brain, a)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (im, title, asp) in zip(axes, views):
        ax.imshow(_colorize(im, cmap), origin="lower", aspect=asp)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("sub-stroke — orthogonal vessel MIPs")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def ortho_mip_compare(gt_hu, pred_hu, brain, zooms, out_path, cmap="viridis"):
    a = float(zooms[2]) / float(zooms[0])
    rows = [("Truth", _ortho_views(gt_hu, brain, a)),
            ("Pred", _ortho_views(pred_hu, brain, a))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for r, (label, views) in enumerate(rows):
        for c, (im, title, asp) in enumerate(views):
            ax = axes[r, c]
            ax.imshow(_colorize(im, cmap), origin="lower", aspect=asp)
            if r == 0:
                ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=13)
    fig.suptitle("Orthogonal vessel MIPs — ground truth (top) vs prediction (bottom)")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _rotated_mip(v, angle, aspect, cmap) -> Image.Image:
    """Rotate about the S-I axis and project along A-P -> (S-I, L-R); Superior at top."""
    rot = ndi.rotate(v, angle, axes=(0, 1), reshape=False, order=1, cval=_BG)
    mip = np.flipud(_norm(rot.max(axis=1)).T)
    img = Image.fromarray(_colorize(mip, cmap))
    if aspect != 1.0:
        img = img.resize((img.width, int(round(img.height * aspect))), Image.BILINEAR)
    return img


def _label(img: Image.Image, text: str, pad: int = 6) -> Image.Image:
    """Draw a title (e.g. 'Truth'/'Pred') centered at the top of a frame, on a dark box."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(16, img.width // 14))  # Pillow >= 10
    except TypeError:
        font = ImageFont.load_default()
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x, y = (img.width - tw) // 2, pad
    draw.rectangle([x - 5, y - 3, x + tw + 5, y + th + 5], fill=(0, 0, 0))
    draw.text((x - l, y - t), text, fill=(255, 255, 255), font=font)
    return img


def _hcat(a: Image.Image, b: Image.Image, gap: int = 10) -> Image.Image:
    h, w = max(a.height, b.height), a.width + gap + b.width
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + gap, 0))
    return canvas


def _masked(vol_hu, brain, downsample):
    v = np.where(brain > 0, vol_hu, _BG).astype(np.float32)
    return v[::downsample, ::downsample, :] if downsample > 1 else v


def _rotation_frames(vols, brain, zooms, n_frames, downsample, cmap, labels=None):
    """vols: list of HU volumes; returns frames (each = the volumes hconcat'd) + angles."""
    vs = [_masked(v, brain, downsample) for v in vols]
    aspect = float(zooms[2]) / (float(zooms[0]) * downsample)
    angles = np.linspace(0, 360, n_frames, endpoint=False)
    frames = []
    for ang in angles:
        imgs = [_rotated_mip(v, ang, aspect, cmap) for v in vs]
        if labels:
            imgs = [_label(im, lab) for im, lab in zip(imgs, labels)]
        frames.append(imgs[0] if len(imgs) == 1 else _hcat(*imgs))
    return frames, angles


def rotating_mip(vols, brain, zooms, gif_path, montage_path, title, n_frames=18,
                 downsample=1, cmap="viridis", duration=200, labels=None):
    frames, angles = _rotation_frames(vols, brain, zooms, n_frames, downsample, cmap, labels)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    pick = np.linspace(0, len(frames), 6, endpoint=False).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, idx in zip(axes.ravel(), pick):
        ax.imshow(np.asarray(frames[idx]))
        ax.set_title(f"{int(angles[idx])}°"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(montage_path, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="0001")
    ap.add_argument("--modality", choices=["cta", "ncct"], default="cta")
    ap.add_argument("--pred", type=Path, default=None,
                    help="predictions dir; if set, render ground truth vs prediction side by side")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap (viridis, magma, gray, ...)")
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS_DIR / "figures3d")
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--duration", type=int, default=200, help="ms per GIF frame (higher = slower)")
    args = ap.parse_args()

    subs = {s.sid: s for s in discover_subjects()}
    if args.sid not in subs:
        raise SystemExit(f"subject {args.sid} not found under {C.DATASET_ROOT}")
    s = subs[args.sid]
    vol, img = load_nii(s.cta if args.modality == "cta" else s.ncct)
    zooms = img.header.get_zooms()[:3]
    ncct_hu, nimg = load_nii(s.ncct)
    brain = get_brain_mask(args.sid, ncct_hu, nimg.affine)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.pred is not None:
        pred_file = C.pred_path(args.sid, args.pred)
        if not pred_file.exists():
            raise SystemExit(f"prediction not found: {pred_file}")
        pred_hu, _ = load_nii(pred_file)
        tag = f"{args.sid}_compare"
        ortho_mip_compare(vol, pred_hu, brain, zooms, args.out / f"ortho_{tag}.png", cmap=args.cmap)
        rotating_mip([vol, pred_hu], brain, zooms, args.out / f"rotate_{tag}.gif",
                     args.out / f"rotate_{tag}_montage.png",
                     title="Rotating vessel MIP — ground truth (left) vs prediction (right)",
                     n_frames=args.frames, downsample=args.downsample, cmap=args.cmap,
                     duration=args.duration, labels=["Truth", "Pred"])
    else:
        tag = f"{args.sid}_{args.modality}"
        ortho_mip(vol, brain, zooms, args.out / f"ortho_{tag}.png", cmap=args.cmap)
        rotating_mip([vol], brain, zooms, args.out / f"rotate_{tag}.gif",
                     args.out / f"rotate_{tag}_montage.png",
                     title="Rotating vessel MIP (about superior-inferior axis)",
                     n_frames=args.frames, downsample=args.downsample, cmap=args.cmap,
                     duration=args.duration)
    print(f"wrote {tag} figures -> {args.out}")


if __name__ == "__main__":
    main()
