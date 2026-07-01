"""Reference baselines that we at least must beat.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

* identity   : copy the NCCT as the CTA prediction. Because NCCT and CTA differ only by
               vascular contrast, this scores well on whole-volume metrics. This the reason
               why MSE alone would be a bad metric.
* hu_remap   : a single global monotonic NCCT-HU -> CTA-HU transfer function fit on the
               training voxels (binned conditional mean), i.e. take the HU, bin it, calculate mean.
               "Naive prediction". Captures global intensity changes without any spatial modelling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .data import Subject, discover_subjects, load_nii, save_hu_int16, split_ids


# --------------------------------------------------------------------------------------
# HU remap (global transfer function)
# --------------------------------------------------------------------------------------
def fit_hu_remap(
    subjects: list[Subject], train_sids: list[str], n_bins: int = 256,
    lo: float = C.PRED_HU_MIN, hi: float = 1500.0, max_voxels: int = 2_000_000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, mapped_hu): the conditional mean CTA HU per NCCT-HU bin."""
    by_id = {s.sid: s for s in subjects}
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    rng = np.random.default_rng(seed)
    per_vol = max(1, max_voxels // max(1, len(train_sids)))
    for sid in train_sids:
        s = by_id[sid]
        ncct, _ = load_nii(s.ncct)
        cta, _ = load_nii(s.cta)
        ncct, cta = ncct.ravel(), cta.ravel() # Not counting for any spatial dependency
        if ncct.size > per_vol:
            sel = rng.choice(ncct.size, per_vol, replace=False)
            ncct, cta = ncct[sel], cta[sel]
        bins = np.clip(np.digitize(ncct, edges) - 1, 0, n_bins - 1)
        np.add.at(sums, bins, cta)
        np.add.at(counts, bins, 1.0)
    mapped = np.where(counts > 0, sums / np.maximum(counts, 1), centers)
    # Fill empty bins by carrying the identity offset so the map stays sensible everywhere.
    empty = counts == 0
    mapped[empty] = centers[empty]
    return centers, mapped.astype(np.float32)


def apply_hu_remap(ncct_hu: np.ndarray, lut: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    centers, mapped = lut
    return np.interp(ncct_hu, centers, mapped).astype(np.float32)


# --------------------------------------------------------------------------------------
# Prediction writers
# --------------------------------------------------------------------------------------
def predict_identity(s: Subject, out_dir: Path) -> Path:
    ncct, img = load_nii(s.ncct)
    out = C.pred_path(s.sid, out_dir)
    save_hu_int16(ncct, img, out)
    return out


def predict_hu_remap(s: Subject, lut, out_dir: Path) -> Path:
    ncct, img = load_nii(s.ncct)
    pred = apply_hu_remap(ncct, lut)
    out = C.pred_path(s.sid, out_dir)
    save_hu_int16(pred, img, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate baseline CTA predictions.")
    ap.add_argument("--method", choices=["identity", "hu_remap"], default="identity")
    ap.add_argument("--out", type=Path, default=C.PREDS_DIR)
    ap.add_argument("--split", type=Path, default=None, help="split.json (for hu_remap fit)")
    args = ap.parse_args()

    subjects = discover_subjects()
    by_id = {s.sid: s for s in subjects}
    test = [by_id[i] for i in C.TEST_IDS if i in by_id]
    if not test:
        raise SystemExit(f"No test subjects found under {C.DATASET_ROOT}. Is the dataset extracted?")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.method == "identity":
        for s in test:
            print("identity ->", predict_identity(s, args.out).name)
    else:
        if args.split and args.split.exists():
            from .data import load_split
            train_sids = load_split(args.split)["train"]
        else:
            train_sids = split_ids(subjects)["train"]
        print(f"Fitting HU remap on {len(train_sids)} training subjects ...")
        lut = fit_hu_remap(subjects, train_sids)
        for s in test:
            print("hu_remap ->", predict_hu_remap(s, lut, args.out).name)


if __name__ == "__main__":
    main()
