"""Qualitative figures: slice panels, error maps, and vessel MIPs.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

A maximum-intensity projection (MIP) over the axial stack is the natural way to read CTA
vasculature, so it is the most informative single view for this task.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import constants as C
from .brain_mask import get_brain_mask
from .data import discover_subjects, load_nii, vessel_mask_hu

_BRAIN_WIN = (0, 80)      # soft-tissue window
_VESSEL_WIN = (0, 400)    # window that shows iodine contrast


def _norm(hu, win):
    return np.clip((hu - win[0]) / (win[1] - win[0]), 0, 1)


def representative_slices(ncct_hu, gt_hu, k=3, axial_axis=2):
    """Pick axial slices with the most enhanced-vessel voxels."""
    vessel = vessel_mask_hu(ncct_hu, gt_hu)
    counts = vessel.sum(axis=tuple(i for i in range(3) if i != axial_axis))
    return list(np.argsort(counts)[::-1][:k])


def save_slice_panel(sid, ncct_hu, gt_hu, pred_hu, brain, out_path, axial_axis=2):
    zs = representative_slices(ncct_hu, gt_hu, k=3, axial_axis=axial_axis)
    nc = np.moveaxis(ncct_hu, axial_axis, 0)
    gt = np.moveaxis(gt_hu, axial_axis, 0)
    pr = np.moveaxis(pred_hu, axial_axis, 0)
    br = np.moveaxis(brain > 0, axial_axis, 0)
    fig, axes = plt.subplots(len(zs), 4, figsize=(12, 3 * len(zs)))
    axes = np.atleast_2d(axes)
    cols = ["NCCT (in)", "CTA (truth)", "CTA (pred)", "|error| in brain (HU)"]
    for i, z in enumerate(zs):
        # Error restricted to the brain: outside it the two acquisitions differ only by
        # skull-edge registration noise, which would otherwise dominate the map.
        err = np.where(br[z], np.abs(pr[z] - gt[z]), 0.0)
        ims = [_norm(nc[z], _VESSEL_WIN), _norm(gt[z], _VESSEL_WIN), _norm(pr[z], _VESSEL_WIN), None]
        for j in range(4):
            ax = axes[i, j]
            if j < 3:
                ax.imshow(ims[j].T, cmap="gray", origin="lower")
            else:
                m = ax.imshow(np.clip(err, 0, 300).T, cmap="magma", origin="lower")
                fig.colorbar(m, ax=ax, fraction=0.046)
            if i == 0:
                ax.set_title(cols[j])
            ax.set_ylabel(f"z={z}" if j == 0 else "")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"sub-stroke{sid}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_mip(sid, gt_hu, pred_hu, brain, out_path, axial_axis=2):
    # Project the MIP *within the brain only*: otherwise the skull (>1000 HU) saturates the
    # projection and hides the intracranial vessels we want to see.
    b = brain > 0
    gt_m = np.where(b, gt_hu, -1000.0)
    pr_m = np.where(b, pred_hu, -1000.0)
    gt_mip = _norm(gt_m.max(axis=axial_axis), _VESSEL_WIN)
    pr_mip = _norm(pr_m.max(axis=axial_axis), _VESSEL_WIN)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    for ax, im, t in zip(axes, [gt_mip, pr_mip], ["CTA MIP (truth)", "CTA MIP (pred)"]):
        ax.imshow(im.T, cmap="gray", origin="lower")
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"sub-stroke{sid} — brain-masked axial MIP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate qualitative figures for test subjects.")
    ap.add_argument("--preds", type=Path, default=C.PREDS_DIR)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS_DIR / "figures")
    ap.add_argument("--subjects", nargs="*", default=None, help="subject ids (default: all test)")
    args = ap.parse_args()

    subjects = {s.sid: s for s in discover_subjects()}
    args.out.mkdir(parents=True, exist_ok=True)
    ids = args.subjects or list(C.TEST_IDS)
    for sid in ids:
        if sid not in subjects:
            continue
        pred_file = C.pred_path(sid, args.preds)
        if not pred_file.exists():
            print(f"  missing prediction for {sid}, skipping")
            continue
        ncct_hu, img = load_nii(subjects[sid].ncct)
        gt_hu, _ = load_nii(subjects[sid].cta)
        pred_hu, _ = load_nii(pred_file)
        brain = get_brain_mask(sid, ncct_hu, img.affine)
        save_slice_panel(sid, ncct_hu, gt_hu, pred_hu.astype(np.float32), brain, args.out / f"panel_{sid}.png")
        save_mip(sid, gt_hu, pred_hu.astype(np.float32), brain, args.out / f"mip_{sid}.png")
        print("figures ->", sid)


if __name__ == "__main__":
    main()
