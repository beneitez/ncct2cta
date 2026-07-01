"""Inference: turn an NCCT volume into a HU CTA prediction and write it to disk.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

3D models we use MONAI sliding-window inference with Gaussian blending. Standard from the MONAI package
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from . import constants as C
from .brain_mask import get_brain_mask
from .data import (
    Subject, apply_windows, discover_subjects, hu_to_scaled, load_nii, load_split,
    n_input_channels, save_hu_int16, scaled_to_hu,
)
from .model import build_model


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cfg_windows(cfg: dict) -> tuple[tuple[float, float], ...]:
    wins = cfg.get("windows", [[C.HU_CLIP_MIN, C.HU_CLIP_MAX]])
    return tuple((float(lo), float(hi)) for lo, hi in wins)


@torch.no_grad()
def predict_residual_3d(model, input_mw: np.ndarray, device, patch, overlap, sw_batch) -> np.ndarray:
    x = torch.from_numpy(input_mw[None]).float().to(device)  # (1, C, H, W, D)
    res = sliding_window_inference(
        x, roi_size=patch, sw_batch_size=sw_batch, predictor=model,
        overlap=overlap, mode="gaussian",
    )
    return res[0, 0].float().cpu().numpy()

def predict_hu(model, ncct_hu: np.ndarray, arch: str, device, cfg: dict, brain=None) -> np.ndarray:
    """Predicted CTA in HU for a full raw-HU NCCT volume (brain-masked residual)."""
    windows = cfg_windows(cfg)
    lo0, hi0 = windows[0]
    input_mw = apply_windows(ncct_hu, windows)  # (C, H, W, D)
    res = predict_residual_3d(
        model, input_mw, device, patch=tuple(cfg.get("patch_size", (192, 192, 32))),
        overlap=cfg.get("overlap", 0.5), sw_batch=cfg.get("sw_batch", 2),
    )
    if brain is not None:
        res = res * (brain > 0)  # synthesize contrast only inside the brain; else CTA == NCCT
    # Add only the in-window contrast delta back onto the TRUE raw NCCT, so HU outside the
    # synthesized region (air, bone, skull) is preserved exactly rather than window-clipped.
    ncct_s = hu_to_scaled(ncct_hu, lo0, hi0)
    recon_s = np.clip(ncct_s + res, -1.0, 1.0)
    delta_hu = scaled_to_hu(recon_s, lo0, hi0) - scaled_to_hu(ncct_s, lo0, hi0)
    return ncct_hu + delta_hu


def predict_subject_to_file(model, s: Subject, arch: str, device, cfg: dict, out_dir: Path) -> Path:
    ncct_hu, img = load_nii(s.ncct)
    brain = get_brain_mask(s.sid, ncct_hu, img.affine, method=cfg.get("brain_method", "auto"))
    pred_hu = predict_hu(model, ncct_hu, arch, device, cfg, brain)
    out = C.pred_path(s.sid, out_dir)
    save_hu_int16(pred_hu, img, out)
    return out


def load_checkpoint(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    in_ch = n_input_channels(len(cfg_windows(cfg)))
    model = build_model(cfg["arch"], in_channels=in_ch)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict CTA for the held-out test subjects.")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=C.PREDS_DIR)
    ap.add_argument("--split", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, cfg = load_checkpoint(args.ckpt, device)

    subjects = discover_subjects()
    by_id = {s.sid: s for s in subjects}
    test_ids = load_split(args.split)["test"] if args.split else list(C.TEST_IDS)
    args.out.mkdir(parents=True, exist_ok=True)
    for sid in test_ids:
        if sid not in by_id:
            print(f"  WARNING: test subject {sid} not found, skipping")
            continue
        out = predict_subject_to_file(model, by_id[sid], cfg["arch"], device, cfg, args.out)
        print("pred ->", out.name)


if __name__ == "__main__":
    main()
