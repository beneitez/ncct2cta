"""Data discovery, train/val split, intensity windowing, and patch/slice datasets.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Design notes
------------
* IO uses nibabel so we can copy the exact affine/header from the NCCT when saving
  predictions.
* It learns a *masked residual* CTA - NCCT. Both volumes are windowed to a common HU range and
  scaled to [-1, 1]; the network predicts the scaled residual, this should be 0 everywhere
  except in contrast-enhanced vessels.
* No resampling to isotropic spacing: the data is anisotropic (according to instruction ~0.4 mm in-plane
  vs ~2 mm through-plane), so we keep spacing and use anisotropic patches 
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from . import constants as C

# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def load_nii(path: str | Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Return (float32 array, nibabel image). The image keeps affine+header for saving."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj, dtype=np.float32)
    return arr, img


def save_hu_int16(arr_hu: np.ndarray, ref_img: nib.Nifti1Image, out_path: str | Path) -> None:
    """Save a HU array as int16 NIfTI, copying affine+header from the reference NCCT."""
    arr = np.clip(np.rint(arr_hu), C.PRED_HU_MIN, C.PRED_HU_MAX).astype(np.int16)
    out = nib.Nifti1Image(arr, ref_img.affine, ref_img.header)
    out.set_data_dtype(np.int16)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))

# --------------------------------------------------------------------------------------
# Subject discovery + split
# --------------------------------------------------------------------------------------
@dataclass(frozen=True) # Define a dataclass for the type of data we work on
class Subject:
    sid: str
    ncct: Path
    cta: Path
    msk: Path | None # Mask might not be available.


def discover_subjects(root: Path | None = None) -> list[Subject]:
    """All subjects that have both ncct and space-ncct cta present, sorted by id."""
    root = root or C.DATASET_ROOT
    raw = root / "raw_data"
    subjects: list[Subject] = [] # Added type of list for type checking
    if not raw.exists():
        return subjects
    for sub_dir in sorted(raw.glob("sub-stroke*")):
        sid = sub_dir.name.replace("sub-stroke", "")
        ncct, cta = C.ncct_path(sid, root), C.cta_path(sid, root)
        if ncct.exists() and cta.exists():
            msk = C.msk_path(sid, root)
            subjects.append(Subject(sid, ncct, cta, msk if msk.exists() else None))
    return subjects


def split_ids(
    subjects: Iterable[Subject], val_frac: float = 0.1, seed: int = 1337
) -> dict[str, list[str]]:
    """Deterministic train/val/test split. The 15 TEST_IDS are always held out."""
    ids = [s.sid for s in subjects]
    test = [i for i in ids if i in C.TEST_IDS]
    pool = sorted(i for i in ids if i not in C.TEST_IDS)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_val = max(1, round(len(pool) * val_frac))
    val, train = pool[:n_val], pool[n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2))


def load_split(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------
# Intensity windowing  (HU <-> scaled [-1, 1])
# --------------------------------------------------------------------------------------
def hu_to_scaled(hu: np.ndarray | float, lo: float = C.HU_CLIP_MIN, hi: float = C.HU_CLIP_MAX):
    x = (np.clip(hu, lo, hi) - lo) / (hi - lo)  # -> [0, 1]
    return x * 2.0 - 1.0  # -> [-1, 1]


def scaled_to_hu(x: np.ndarray | torch.Tensor, lo: float = C.HU_CLIP_MIN, hi: float = C.HU_CLIP_MAX):
    x01 = (x + 1.0) / 2.0
    return x01 * (hi - lo) + lo


# Default window set: a single wide window that preserves brain + vessel contrast.
# We can pass extra windows (e.g. a brain window) to give the model multiple clinically-windowed
# views of the same NCCT, i.e. providing more information
# the ISLES'24 winners' insight, cf paper in papers.
# The first window is the "primary": it defines the residual/reconstruction space.

DEFAULT_WINDOWS: tuple[tuple[float, float], ...] = ((C.HU_CLIP_MIN, C.HU_CLIP_MAX),)

def apply_windows(arr_hu: np.ndarray, windows) -> np.ndarray:
    """Stack scaled views of one HU array, one channel per window: (n_windows, *arr.shape)."""
    return np.stack([hu_to_scaled(arr_hu, lo, hi) for (lo, hi) in windows], axis=0).astype(np.float32)


def vessel_mask_hu(ncct_hu: np.ndarray, cta_hu: np.ndarray) -> np.ndarray:
    """Enhanced-vessel mask: bright in CTA but soft tissue in NCCT (excludes bone)."""
    return (cta_hu > C.VESSEL_CTA_HU) & (ncct_hu < C.VESSEL_NCCT_HU) # Only iodine contrast


def brain_mask_hu(ncct_hu: np.ndarray) -> np.ndarray:
    """Coarse intracranial soft-tissue mask from NCCT (used only for loss weighting)."""
    return (ncct_hu > -20.0) & (ncct_hu < 100.0) # Roughly where the brain is, used only if no brain mask provided


def vessel_weight_map(
    ncct_s: torch.Tensor, cta_s: torch.Tensor, brain: torch.Tensor | None = None,
    w_vessel: float = 20.0, w_brain: float = 1.0, w_bg: float = 0.1,
) -> torch.Tensor:
    """Per-voxel loss weights from primary-window scaled tensors.

    `brain` (bool tensor) is the intracranial mask; if omitted we use
    a crude HU-threshold brain. Vessels = bright in CTA but not highlighted in NCCT, inside
    the brain.
    """
    cta_thr = float(hu_to_scaled(C.VESSEL_CTA_HU))
    ncct_hi = float(hu_to_scaled(C.VESSEL_NCCT_HU))
    if brain is None:
        ncct_lo = float(hu_to_scaled(-20.0))
        brain = (ncct_s > ncct_lo) & (ncct_s < ncct_hi)
    else:
        brain = brain > 0
    vessel = (cta_s > cta_thr) & (ncct_s < ncct_hi) & brain
    w = torch.full_like(ncct_s, w_bg)
    w[brain] = w_brain
    w[vessel] = w_vessel
    return w


# --------------------------------------------------------------------------------------
# Volume cache (optional in-RAM)
# --------------------------------------------------------------------------------------
class VolumeStore:
    """Loads raw-HU (ncct, cta) volumes + an intracranial brain mask, optionally cached.

    Returns raw HU (not pre-scaled) so the datasets can build multiple windowed input
    channels on the fly. The brain mask is computed once and disk-cached
    by `brain_mask.get_brain_mask`.
    """

    def __init__(self, subjects: dict[str, Subject], cache: bool = False, brain_method: str = "auto"):
        self._subjects = subjects
        self._cache_on = cache
        self._brain_method = brain_method
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(self, sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sid in self._cache:
            nc, ct, br = self._cache[sid]
            return nc.astype(np.float32), ct.astype(np.float32), br.astype(np.float32)
        from .brain_mask import get_brain_mask  # local import only if needed
        s = self._subjects[sid]
        ncct_hu, nimg = load_nii(s.ncct)
        cta_hu, _ = load_nii(s.cta)
        brain = get_brain_mask(sid, ncct_hu, nimg.affine, method=self._brain_method)
        if self._cache_on:
            # Cache compactly: CT HU is integer -> int16 (lossless), mask -> uint8.
            # ~2.4x less RAM than float32 so the full train set fits (~15-20 GB, not ~40+).
            self._cache[sid] = (
                np.clip(np.rint(ncct_hu), -32768, 32767).astype(np.int16),
                np.clip(np.rint(cta_hu), -32768, 32767).astype(np.int16),
                brain.astype(np.uint8),
            )
        return ncct_hu.astype(np.float32), cta_hu.astype(np.float32), brain.astype(np.float32)


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class PatchDataset3D(Dataset):
    """Random anisotropic 3D patches. Multi-window NCCT input + primary-window ref/target."""

    def __init__(
        self, sids: list[str], store: VolumeStore,
        patch_size: tuple[int, int, int] = (192, 192, 32), samples_per_volume: int = 4,
        seed: int = 0, windows=DEFAULT_WINDOWS,
    ):
        self.sids = sids
        self.store = store
        self.patch = patch_size
        self.spv = samples_per_volume
        self.windows = windows
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.sids) * self.spv

    def _rand_crop(self, arrs: list[np.ndarray]):
        ph, pw, pd = self.patch # Patch dims
        H, W, D = arrs[0].shape # Current vol dims
        pad = [(0, max(0, p - s)) for p, s in zip(self.patch, (H, W, D))] # return 0 if dims big enough
        if any(b for _, b in pad):
            # Just replicate if we run out of volume to avoid issues with changing matter in HU units
            arrs = [np.pad(a, pad, mode="edge") for a in arrs] 
            H, W, D = arrs[0].shape
        z0 = int(self.rng.integers(0, H - ph + 1))
        y0 = int(self.rng.integers(0, W - pw + 1))
        x0 = int(self.rng.integers(0, D - pd + 1))
        sl = (slice(z0, z0 + ph), slice(y0, y0 + pw), slice(x0, x0 + pd))
        return [a[sl] for a in arrs]

    def __getitem__(self, idx: int):
        sid = self.sids[idx // self.spv]
        ncct_hu, cta_hu, brain = self.store.get(sid)
        nc, ct, br = self._rand_crop([ncct_hu, cta_hu, brain])
        lo0, hi0 = self.windows[0]
        return {
            "input": torch.from_numpy(apply_windows(nc, self.windows)),       # (C, H, W, D)
            "ncct": torch.from_numpy(hu_to_scaled(nc, lo0, hi0)[None].astype(np.float32)),
            "cta": torch.from_numpy(hu_to_scaled(ct, lo0, hi0)[None].astype(np.float32)),
            "brain": torch.from_numpy(br[None].astype(np.float32)),
            "sid": sid,
        }

def n_input_channels(n_windows: int) -> int:
    return n_windows


def build_volume_store(
    subjects: list[Subject], sids: list[str], cache: bool, brain_method: str = "auto"
) -> VolumeStore:
    by_id = {s.sid: s for s in subjects}
    return VolumeStore(
        {sid: by_id[sid] for sid in sids if sid in by_id}, cache=cache, brain_method=brain_method)
