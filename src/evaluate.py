"""Evaluate CTA predictions against ground truth on the held-out test subjects.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Because NCCT and CTA are almost identical outside vessels, whole-volume metrics are dominated
by the trivial background. 

Every metric is reported in three regions: whole / brain / vessel) and compare every 
model against the identity baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import constants as C
from .brain_mask import get_brain_mask
from .data import discover_subjects, load_nii, vessel_mask_hu

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

_WIN = (C.HU_CLIP_MIN, C.HU_CLIP_MAX)
_RANGE = _WIN[1] - _WIN[0]


def _win01(hu: np.ndarray) -> np.ndarray:
    return (np.clip(hu, *_WIN) - _WIN[0]) / _RANGE


def mae(a, b, m):
    return float(np.abs(a[m] - b[m]).mean()) if m.sum() else float("nan")


def rmse(a, b, m):
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if m.sum() else float("nan")


def psnr(a, b, m):
    if not m.sum():
        return float("nan")
    mse = ((np.clip(a, *_WIN)[m] - np.clip(b, *_WIN)[m]) ** 2).mean()
    return float(20 * np.log10(_RANGE) - 10 * np.log10(mse + 1e-8))


def ssim_slicewise(pred_hu, gt_hu, axial_axis=2):
    if not _HAS_SK:
        return float("nan")
    p, g = _win01(pred_hu), _win01(gt_hu)
    p = np.moveaxis(p, axial_axis, 0)
    g = np.moveaxis(g, axial_axis, 0)
    vals = [_ssim(g[z], p[z], data_range=1.0) for z in range(g.shape[0])]
    return float(np.mean(vals))


def evaluate_subject(pred_hu, ncct_hu, gt_hu, brain) -> dict:
    whole = np.ones_like(gt_hu, dtype=bool)
    brain = brain > 0
    vessel = vessel_mask_hu(ncct_hu, gt_hu) & brain
    row = {
        "mae_whole": mae(pred_hu, gt_hu, whole),
        "mae_brain": mae(pred_hu, gt_hu, brain),
        "mae_vessel": mae(pred_hu, gt_hu, vessel),
        "rmse_vessel": rmse(pred_hu, gt_hu, vessel),
        "psnr_brain": psnr(pred_hu, gt_hu, brain),
        "ssim": ssim_slicewise(pred_hu, gt_hu),
        "n_vessel_vox": int(vessel.sum()),
    }
    return row

def evaluate_preds(label: str, preds_dir: Path, subjects_by_id) -> list[dict]:
    rows = []
    for sid in C.TEST_IDS:
        if sid not in subjects_by_id:
            continue
        pred_file = C.pred_path(sid, preds_dir)
        if not pred_file.exists():
            print(f"  [{label}] missing prediction for {sid}, skipping")
            continue
        s = subjects_by_id[sid]
        ncct_hu, img = load_nii(s.ncct)
        gt_hu, _ = load_nii(s.cta)
        pred_hu, _ = load_nii(pred_file)
        brain = get_brain_mask(sid, ncct_hu, img.affine)
        r = evaluate_subject(pred_hu.astype(np.float32), ncct_hu, gt_hu, brain)
        r.update({"label": label, "sid": sid})
        rows.append(r)
    return rows


def summarize(rows: list[dict]):
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = ["mae_whole", "mae_brain", "mae_vessel", "rmse_vessel", "psnr_brain", "ssim"]
    agg = df.groupby("label")[metrics].agg(["mean", "std"])
    return df, agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate one or more prediction sets.")
    ap.add_argument("--pred", action="append", default=[], metavar="LABEL=DIR",
                    help="e.g. --pred identity=preds_identity --pred unet=preds")
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS_DIR / "eval")
    args = ap.parse_args()
    if not args.pred:
        raise SystemExit("Provide at least one --pred LABEL=DIR")

    subjects = discover_subjects()
    by_id = {s.sid: s for s in subjects}
    all_rows = []
    for spec in args.pred:
        label, _, d = spec.partition("=")
        all_rows += evaluate_preds(label, Path(d), by_id)
    if not all_rows:
        raise SystemExit("No predictions evaluated.")

    df, agg = summarize(all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_subject.csv", index=False)
    agg.to_csv(args.out / "summary.csv")
    try:
        md = agg.round(3).to_markdown()
    except ImportError:  # tabulate not installed
        md = "```\n" + agg.round(3).to_string() + "\n```"
    (args.out / "summary.md").write_text(md)
    print("\n=== Summary (mean over test subjects) ===")
    print(agg.round(3).to_string())
    print(f"\nWrote {args.out/'per_subject.csv'} and {args.out/'summary.md'}")


if __name__ == "__main__":
    main()
