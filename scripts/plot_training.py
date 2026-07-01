"""Plot training curves from a train_log.csv (epoch, train_loss, val_vessel_mae_hu, secs).


Miguel Beneitez - beneitez@protonmail.com

AI Disclaimer: 
The code initial prototype was structured using Claude Code
human-implemented functions were tidied up with Claude Code.
All coding decisions, code revision and feature implementations
are human-handled. 

References: Ren at al. Proc. Machinbe Learning Research 2025 and those therein

Run:  python -m scripts.plot_training --log artifacts/unet3d/train_log.csv \
          --out artifacts/unet3d/training_curves.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.log.with_suffix(".png")

    df = pd.read_csv(args.log)
    val = df.dropna(subset=["val_vessel_mae_hu"])
    best = val.loc[val["val_vessel_mae_hu"].idxmin()] if len(val) else None

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    # (a) train loss
    ax[0].plot(df["epoch"], df["train_loss"], color="#1f77b4", lw=1.5)
    ax[0].set(title="Training loss (vessel-weighted L1)", xlabel="epoch", ylabel="loss")
    ax[0].grid(alpha=0.3)

    # (b) validation vessel MAE (HU)
    ax[1].plot(val["epoch"], val["val_vessel_mae_hu"], "o-", color="#d62728", lw=1.5)
    if best is not None:
        ax[1].scatter([best["epoch"]], [best["val_vessel_mae_hu"]], s=140,
                      facecolors="none", edgecolors="k", zorder=5,
                      label=f"best: {best['val_vessel_mae_hu']:.1f} HU @ ep {int(best['epoch'])}")
        ax[1].legend()
    ax[1].set(title="Validation vessel MAE (HU)  ↓", xlabel="epoch", ylabel="MAE (HU)")
    ax[1].grid(alpha=0.3)

    # (c) seconds per epoch
    ax[2].plot(df["epoch"], df["secs"], color="#2ca02c", lw=1.5)
    ax[2].set(title="Wall-clock per epoch (s)", xlabel="epoch", ylabel="seconds")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    if best is not None:
        print(f"best val vessel-MAE = {best['val_vessel_mae_hu']:.2f} HU at epoch {int(best['epoch'])}")
        print(f"last train_loss = {df['train_loss'].iloc[-1]:.4f} (epoch {int(df['epoch'].iloc[-1])})")
        print(f"median epoch time (cached) = {df['secs'].iloc[1:].median():.1f}s")


if __name__ == "__main__":
    main()
