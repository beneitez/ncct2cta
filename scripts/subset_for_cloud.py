"""Copy only the files this task needs (NCCT + space-ncct CTA, optional mask) into a
compact BIDS-style folder for upload to a cloud GPU. Avoids shipping the large CTP/4D
modalities.

Run:  python -m scripts.subset_for_cloud --out /path/to/ISLES2024_subset [--with-mask]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import constants as C
from src.data import discover_subjects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-mask", action="store_true", help="also copy train-time vessel masks")
    args = ap.parse_args()

    subjects = discover_subjects()
    n_files = 0
    total_bytes = 0
    for s in subjects:
        items = [(s.ncct, C.NCCT_REL), (s.cta, C.CTA_REL)]
        if args.with_mask and s.msk is not None:
            items.append((s.msk, C.MSK_REL))
        for src_path, rel in items:
            dst = args.out / rel.format(sid=s.sid)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            n_files += 1
            total_bytes += dst.stat().st_size
    print(f"Copied {n_files} files for {len(subjects)} subjects "
          f"({total_bytes/1e9:.2f} GB) -> {args.out}")
    print("On the cloud box:  export ISLES_ROOT=", args.out, sep="")


if __name__ == "__main__":
    main()
