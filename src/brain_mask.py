"""Brain extraction (skull-stripping), following the ISLES'24 winning solution.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Ren et al. credit SynthStrip skull-stripping + custom intensity windowing for their first place. We use the
brain mask to (a) weight the loss toward intracranial tissue, (b) restrict the synthesized
residual to the brain (outside the brain CTA == NCCT), and (c) report brain/vessel metrics.

Mask source is pluggable and disk-cached per subject:
  * "synthstrip" : SynthStrip could be used, used by the winners of the challenge.
  * "morph"      : dependency-light morphological CT brain extraction (scipy/skimage).
  * "auto"       : SynthStrip if available (NOT IMPLEMENTED), else "morph".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from . import constants as C

_CACHE = C.ARTIFACTS_DIR / "brain_masks"


# --------------------------------------------------------------------------------------
# Availability / method resolution
# --------------------------------------------------------------------------------------

def resolve_method(method: str) -> str:
    if method == "auto":
        return "morph"
    return method


# --------------------------------------------------------------------------------------
# SynthStrip
# --------------------------------------------------------------------------------------
# TBImplemented

# --------------------------------------------------------------------------------------
# Morphological CT brain extraction (fallback)
# --------------------------------------------------------------------------------------
def _morph_ct_brain(ncct_hu: np.ndarray) -> np.ndarray:
    """Approximate intracranial mask: soft tissue, largest 3D component, fill, erode.

    Light-dependent brain identification for loss weighting and region-restricted eval.
    """
    soft = (ncct_hu >= -20) & (ncct_hu <= 140)        # brain/CSF/blood, excludes bone & air
    soft = ndimage.binary_opening(soft, iterations=2)  # break thin scalp/skull bridges
    lab, n = ndimage.label(soft)
    if n == 0:
        return np.zeros_like(soft, dtype=bool)
    sizes = ndimage.sum(np.ones_like(lab, dtype=np.float32), lab, index=range(1, n + 1))
    brain = lab == (int(np.argmax(sizes)) + 1)         # largest component ~ intracranial
    brain = ndimage.binary_fill_holes(brain)           # fill ventricles/vessels
    brain = ndimage.binary_erosion(brain, iterations=2)  # step inside the skull
    return brain


# --------------------------------------------------------------------------------------
# Public API (disk-cached)
# --------------------------------------------------------------------------------------
def _save_atomic(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write the mask atomically: save to a unique temp file, then os.replace into place.

    With multiple DataLoader workers this prevents a reader from ever seeing a half-written
    .nii.gz (which would raise EOFError). os.replace is atomic on a single filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"._tmp{os.getpid()}_{path.name}")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(tmp))
    os.replace(tmp, path)


def get_brain_mask(
    sid: str, ncct_hu: np.ndarray, affine: np.ndarray, method: str = "auto",
    cache_dir: Path | None = None,
) -> np.ndarray:
    cache_dir = cache_dir or _CACHE
    m = resolve_method(method)
    path = cache_dir / f"{sid}_{m}.nii.gz"
    if path.exists():
        try:
            return np.asanyarray(nib.load(str(path)).dataobj) > 0
        except Exception:
            pass  # corrupt/partial cache (e.g. an interrupted write) -> recompute below
    if m == "synthstrip":
        try:
            pass # Not implemented
        except Exception as e:  # robust: never let a missing tool break training
            print(f"  [brain_mask] SynthStrip failed for {sid} ({e}); using morph fallback")
            m, mask = "morph", _morph_ct_brain(ncct_hu)
            path = cache_dir / f"{sid}_{m}.nii.gz"
    else:
        mask = _morph_ct_brain(ncct_hu)
    _save_atomic(mask, affine, path)
    return mask.astype(bool)
