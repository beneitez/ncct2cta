"""Pre-compute and cache all brain masks single-process, pre-training.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

On a cluster the DataLoader spawns several worker processes; if the brain-mask cache is
empty they all compute & write the same files at once on the first epoch, which both
duplicates work and can corrupt a .nii.gz mid-write (EOFError). Running this once first
means the workers only ever READ the cache.

Run after extraction, before sbatch training:
    python -m scripts.precompute_brain_masks --method auto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import constants as C
from src.brain_mask import get_brain_mask, resolve_method
from src.data import discover_subjects, load_nii


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="auto", choices=["auto", "synthstrip", "morph"])
    args = ap.parse_args()

    subjects = discover_subjects()
    if not subjects:
        raise SystemExit(f"No subjects under {C.DATASET_ROOT}; set ISLES_ROOT.")
    print(f"Computing '{resolve_method(args.method)}' brain masks for {len(subjects)} "
          f"subjects -> {C.ARTIFACTS_DIR / 'brain_masks'}")
    for i, s in enumerate(subjects, 1):
        ncct_hu, img = load_nii(s.ncct)
        get_brain_mask(s.sid, ncct_hu, img.affine, method=args.method)
        print(f"  [{i:3d}/{len(subjects)}] {s.sid}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
