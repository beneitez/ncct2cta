"""Config-driven training for NCCT->CTA residual synthesis.

Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Device is auto-selected (cuda > mps > cpu).
Validation reports vessel-restricted MAE in HU on held-out *val*
subjects and the best checkpoint is kept by that metric. Note, others could be chosen

To run training:
    python -m src.train --config configs/unet3d_cloud.yaml --out artifacts/unet3d
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import constants as C
from .brain_mask import get_brain_mask
from .data import (
    PatchDataset3D, build_volume_store, discover_subjects, load_nii,
    n_input_channels, save_split, split_ids, vessel_mask_hu,
)
from .infer import pick_device, predict_hu
from .losses import SynthesisLoss
from .model import build_model

DEFAULTS = dict(
    arch="unet3d", patch_size=[192, 192, 32], samples_per_volume=4,
    batch_size=2, epochs=100, lr=2e-4, weight_decay=1e-5, w_vessel=20.0, lambda_ssim=0.0,
    cache=False, val_interval=5, max_val_subjects=8, num_workers=0, seed=1337,
    max_train_subjects=0, val_frac=0.1, overlap=0.5, sw_batch=2,
    windows=[[-100, 600]], brain_method="auto",
)


def as_windows(cfg) -> tuple[tuple[float, float], ...]:
    return tuple((float(lo), float(hi)) for lo, hi in cfg["windows"])


def load_config(path: Path | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(model, subjects_by_id, val_ids, arch, device, cfg) -> float:
    """Mean vessel-region MAE (HU) over up to max_val_subjects val volumes."""
    model.eval()
    maes, n_empty, n_nan = [], 0, 0
    for sid in val_ids[: cfg["max_val_subjects"]]:
        s = subjects_by_id[sid]
        ncct_hu, img = load_nii(s.ncct)
        cta_hu, _ = load_nii(s.cta)
        brain = get_brain_mask(sid, ncct_hu, img.affine, method=cfg["brain_method"])
        pred_hu = predict_hu(model, ncct_hu, arch, device, cfg, brain)
        mask = vessel_mask_hu(ncct_hu, cta_hu) & (brain > 0)
        if mask.sum() == 0:
            n_empty += 1
            continue
        mae = float(np.abs(pred_hu[mask] - cta_hu[mask]).mean())
        if mae != mae:  # NaN prediction
            n_nan += 1
            continue
        maes.append(mae)
    if not maes:  # surface the reason instead of silently returning NaN
        print(f"  [validate] no usable val subjects (empty-mask={n_empty}, nan-pred={n_nan}). "
              f"Check brain_method/mask cache and vessel HU thresholds.")
        return float("nan")
    return float(np.mean(maes))


def build_loader(arch, sids, store, cfg):
    windows = as_windows(cfg)
    ds = PatchDataset3D(
        sids, store, patch_size=tuple(cfg["patch_size"]),
        samples_per_volume=cfg["samples_per_volume"], seed=cfg["seed"], windows=windows,
    )
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
        drop_last=True, pin_memory=False,
        persistent_workers=cfg["num_workers"] > 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NCCT->CTA residual model.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=C.ARTIFACTS_DIR)
    ap.add_argument("--device", default="auto")
    # common overrides
    ap.add_argument("--arch", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-subjects", type=int, default=None, dest="max_train_subjects")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "arch": args.arch, "epochs": args.epochs,
        "max_train_subjects": args.max_train_subjects,
    })
    set_seed(cfg["seed"])
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  arch={cfg['arch']}  epochs={cfg['epochs']}  batch={cfg['batch_size']}  "
          f"lr={cfg['lr']}  w_vessel={cfg['w_vessel']}  patch={cfg['patch_size']}")

    subjects = discover_subjects()
    if not subjects:
        raise SystemExit(f"No subjects found under {C.DATASET_ROOT}. Extract the dataset first.")
    by_id = {s.sid: s for s in subjects}
    split = split_ids(subjects, val_frac=cfg["val_frac"], seed=cfg["seed"])
    if cfg["max_train_subjects"]:
        split["train"] = split["train"][: cfg["max_train_subjects"]]
        split["val"] = split["val"][: max(1, cfg["max_val_subjects"])]
    save_split(split, args.out / "split.json")
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    store = build_volume_store(subjects, split["train"], cache=cfg["cache"], brain_method=cfg["brain_method"])
    loader = build_loader(cfg["arch"], split["train"], store, cfg)

    in_ch = n_input_channels(len(as_windows(cfg)))
    print(f"in_channels={in_ch}  windows={as_windows(cfg)}  brain_method={cfg['brain_method']}")
    model = build_model(cfg["arch"], in_channels=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    criterion = SynthesisLoss(
        spatial_dims=3, w_vessel=cfg["w_vessel"], lambda_ssim=cfg["lambda_ssim"],
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = args.out / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_vessel_mae_hu", "secs"])

    best = float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in loader:
            ncct = batch["ncct"].to(device)
            cta = batch["cta"].to(device)
            inp = batch["input"].to(device)
            brain = batch["brain"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred_res = model(inp)
                out = criterion(pred_res, ncct, cta, brain)
            scaler.scale(out["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(out["loss"].detach()))
        sched.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")

        # Validate on epoch 1 (early signal), every val_interval, and the last epoch.
        do_val = epoch == 1 or epoch % cfg["val_interval"] == 0 or epoch == cfg["epochs"]
        val_mae = validate(model, by_id, split["val"], cfg["arch"], device, cfg) if do_val else None
        if do_val and val_mae == val_mae and val_mae < best:  # not NaN and improved
            best = val_mae
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val_mae": val_mae},
                args.out / "best.pt",
            )
        dt = time.time() - t0
        val_str = "  -- (not this epoch)" if not do_val else f"{val_mae:.2f}"
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_vessel_mae_hu={val_str}  ({dt:.1f}s)")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", "" if not do_val else f"{val_mae:.4f}", f"{dt:.1f}"])

    torch.save({"model": model.state_dict(), "config": cfg, "epoch": cfg["epochs"]}, args.out / "last.pt")
    print(f"done. best val vessel-MAE(HU)={best:.2f}  artifacts in {args.out}")


if __name__ == "__main__":
    main()
