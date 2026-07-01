"""Project-wide constants: paths, the fixed test split, and HU windows.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
further implemented functions were cleaned up with Claude code.
All coding decisions, code revision and feature implementations
are human-handled. 

The dataset root can be overridden with the ISLES_ROOT environment variable so the
same code runs unchanged on the local M1 machine and on a cloud GPU box.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
# Default assumes the extracted ISLES2024 BIDS root sits next to this project. Override
# with:  export ISLES_ROOT=/path/to/ISLES2024
_DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "ISLES2024" / "train"
DATASET_ROOT = Path(os.environ.get("ISLES_ROOT", _DEFAULT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDS_DIR = Path(os.environ.get("PREDS_DIR", PROJECT_ROOT / "preds"))
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))

# Path templates (relative to DATASET_ROOT). {sid} is the 4-digit zero-padded id.
NCCT_REL = "raw_data/sub-stroke{sid}/ses-01/sub-stroke{sid}_ses-01_ncct.nii.gz"
CTA_REL = "derivatives/sub-stroke{sid}/ses-01/sub-stroke{sid}_ses-01_space-ncct_cta.nii.gz"

# Vessel-occlusion / mask: train-time only (NEVER available for the held-out test).
# Masks are used to improve training
MSK_REL = "derivatives/sub-stroke{sid}/ses-01/sub-stroke{sid}_ses-01_space-ncct_msk.nii.gz"

# Output filename template. Matches task requirements
PRED_NAME = "sub-stroke{sid}_ses-01_space-ncct_cta_pred.nii.gz"

# --------------------------------------------------------------------------------------
# Fixed held-out test split.
# These 15 subjects must never be seen during training or validation.
# --------------------------------------------------------------------------------------

TEST_IDS: tuple[str, ...] = (
    "0001", "0011", "0022", "0040", "0057",
    "0077", "0087", "0097", "0107", "0117",
    "0140", "0150", "0161", "0171", "0181",
)

# --------------------------------------------------------------------------------------
# Intensity windows (Hounsfield Units)
# --------------------------------------------------------------------------------------
# Key aspect: preprocessing greatly affects the predictions. We normalise the NCCT/CTA
# before the networks. Chosen to preserve:
# - brain tissue (~0-80 HU) 
# - iodine contrast band in vessels (~150-450 HU).
# - Bone (>~600 HU) saturates, which is harmless because it is identical in NCCT and CTA (residual ~ 0).
# Some leeway is allow to not oveconstrict the results.
HU_CLIP_MIN = -100.0
HU_CLIP_MAX = 600.0

# A voxel is treated as "highlighted vessel" when it is bright in the CTA but was soft
# tissue in the NCCT (due to iodine added contrast). This excludes bone, which is bright (saturated)
# in both modalities. Used for loss weighting and vessel-restricted evaluation.
VESSEL_CTA_HU = 150.0   # CTA brighter than this min iodine contrast
VESSEL_NCCT_HU = 100.0  # NCCT below this, unenhances blood ~20-90 HU (was not already bone/calcium)

# Plausible HU range to clip final predictions.
PRED_HU_MIN = -1024
PRED_HU_MAX = 3071


def ncct_path(sid: str, root: Path | None = None) -> Path:
    return (root or DATASET_ROOT) / NCCT_REL.format(sid=sid)


def cta_path(sid: str, root: Path | None = None) -> Path:
    return (root or DATASET_ROOT) / CTA_REL.format(sid=sid)


def msk_path(sid: str, root: Path | None = None) -> Path:
    return (root or DATASET_ROOT) / MSK_REL.format(sid=sid)


def pred_path(sid: str, out_dir: Path | None = None) -> Path:
    return (out_dir or PREDS_DIR) / PRED_NAME.format(sid=sid)
