"""Hard format checks on a predictions directory before submission.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Asserts: exactly 15 files with the required names; each is int16; each matches its
subject's NCCT shape and affine; values lie in a plausible HU range and cover the volume.

Run:  python -m scripts.verify_preds --preds preds/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np

from src import constants as C
from src.data import discover_subjects, load_nii


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=C.PREDS_DIR)
    args = ap.parse_args()

    subjects = {s.sid: s for s in discover_subjects()}
    problems: list[str] = []
    present = 0

    for sid in C.TEST_IDS:
        f = C.pred_path(sid, args.preds)
        if not f.exists():
            problems.append(f"{sid}: MISSING file {f.name}")
            continue
        present += 1
        img = nib.load(str(f))
        dtype = img.get_data_dtype()
        if dtype != np.int16:
            problems.append(f"{sid}: dtype is {dtype}, expected int16")
        if sid in subjects:
            ncct, nimg = load_nii(subjects[sid].ncct)
            if img.shape != ncct.shape:
                problems.append(f"{sid}: shape {img.shape} != NCCT {ncct.shape}")
            if not np.allclose(img.affine, nimg.affine, atol=1e-3):
                problems.append(f"{sid}: affine differs from NCCT")
        arr = np.asanyarray(img.dataobj)
        if arr.min() < C.PRED_HU_MIN - 1 or arr.max() > C.PRED_HU_MAX + 1:
            problems.append(f"{sid}: HU out of range [{arr.min()}, {arr.max()}]")

    extra = [p.name for p in args.preds.glob("*.nii.gz")
             if p.name not in {C.PRED_NAME.format(sid=s) for s in C.TEST_IDS}]
    if extra:
        problems.append(f"Unexpected extra files: {extra}")

    print(f"Predictions dir: {args.preds}")
    print(f"Expected 15 files, found {present}.")
    if problems:
        print("\nFAILED checks:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("All format checks passed. Ready to submit.")


if __name__ == "__main__":
    main()
