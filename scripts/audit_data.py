"""Audit the extracted ISLES2024 dataset before training.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Checks: 
- all 149 ncct/cta pairs present
- per-subject shape & affine match between input and target
- voxel spacing (confirms anisotropy)
- HU ranges for NCCT vs CTA (locates the contrast band)
- presence of all 15 test subjects. 

Writes a per-subject CSV summary used to choose patch sizes and the HU window.

Run from the project root:  python -m scripts.audit_data
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure we have access to all the "aipek" packages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import constants as C
from src.data import discover_subjects, load_nii, vessel_mask_hu

def main() -> None:
    subjects = discover_subjects()
    print(f"DATASET_ROOT = {C.DATASET_ROOT}")
    print(f"Found {len(subjects)} subjects with both NCCT and CTA.")
    if not subjects:
        raise SystemExit("Nothing found: check dataset extracted and ISLES_ROOT")

    ids = {s.sid for s in subjects}
    missing_test = [t for t in C.TEST_IDS if t not in ids]
    print(f"Test subjects present: {len(C.TEST_IDS) - len(missing_test)}/15"
          + (f"  MISSING: {missing_test}" if missing_test else "  (all present)"))

    rows = []
    for s in subjects:
        ncct, nimg = load_nii(s.ncct)
        cta, cimg = load_nii(s.cta)
        zooms = tuple(round(float(z), 3) for z in nimg.header.get_zooms()[:3])
        shape_ok = ncct.shape == cta.shape
        affine_ok = np.allclose(nimg.affine, cimg.affine, atol=1e-3)
        vessel = vessel_mask_hu(ncct, cta)
        rows.append({
            "sid": s.sid, "is_test": s.sid in C.TEST_IDS, "has_msk": s.msk is not None,
            "shape": "x".join(map(str, ncct.shape)), "spacing_mm": zooms,
            "shape_match": shape_ok, "affine_match": affine_ok,
            "ncct_p1": round(float(np.percentile(ncct, 1)), 1),
            "ncct_p99": round(float(np.percentile(ncct, 99)), 1),
            "cta_p1": round(float(np.percentile(cta, 1)), 1),
            "cta_p99": round(float(np.percentile(cta, 99)), 1),
            "cta_max": round(float(cta.max()), 1),
            "vessel_vox": int(vessel.sum()),
            "vessel_frac_pct": round(100 * vessel.mean(), 3),
        })

    df = pd.DataFrame(rows)
    out = C.ARTIFACTS_DIR / "data_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    bad_shape = df.loc[~df.shape_match, "sid"].tolist()
    bad_affine = df.loc[~df.affine_match, "sid"].tolist()
    print(f"\nShape mismatches:  {bad_shape or 'none'}")
    print(f"Affine mismatches: {bad_affine or 'none'}")
    print(f"\nVessel fraction (%) — mean {df.vessel_frac_pct.mean():.3f}, "
          f"median {df.vessel_frac_pct.median():.3f}  "
          f"(confirms vessels are a small fraction of voxels)")
    print(f"CTA p99 HU — mean {df.cta_p99.mean():.0f} (contrast band)")
    print("Unique spacings:", sorted(df.spacing_mm.unique().tolist())[:8], "...")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
