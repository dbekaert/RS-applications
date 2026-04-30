"""
OPERA RTC-S1 fire-scar detection — Artillerie Schietkamp, 't Harde.

A wildfire broke out on 29 April 2026 at the military artillery range
near 't Harde, Gelderland.  This script fetches OPERA RTC-S1 data for
a few passes before and after the fire and produces:

  1. RTC colour composites (before / after)
  2. A VV / VH change map to highlight the burn scar

Usage:
    conda activate RS-applications
    python run_tharde_fire.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Ensure rs_tools is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, LoadedItem
from rs_tools.visualization.rtc_composite import rtc_composite


# ── output directory ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── area of interest ─────────────────────────────────────────────────────
# Tight crop around the Artillerie Schietkamp, 't Harde, Gelderland.
bbox = BoundingBox(west=5.72, south=52.32, east=5.90, north=52.41)

# ── temporal range ───────────────────────────────────────────────────────
# Fire broke out 29 April 2026 — grab a month before and a few days after.
START_DATE = "2026-04-01"
END_DATE   = "2026-05-15"


# ════════════════════════════════════════════════════════════════════════════
# 1. Load OPERA RTC-S1 data (Terrascope)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Fetching OPERA RTC-S1 data around 't Harde …")
print("=" * 60)

data = load_dataset(
    "OPERA_RTC_S1",
    bbox=bbox,
    start_date=START_DATE,
    end_date=END_DATE,
    archive="terrascope",
    assets=["VV", "VH"],
    limit=50,
    mosaic=True,
    output_dir=os.path.join(OUT_DIR, "passes"),
)

print(f"\nLoaded {len(data)} passes:")
for d in data:
    print(f"  {d.label}")

if len(data) < 2:
    print("ERROR: need at least 2 passes (before + after fire). Exiting.")
    sys.exit(1)

# ── Split into before / after fire (29 April 2026) ─────────────────────
from datetime import date as _date
FIRE_DATE = _date(2026, 4, 29)

before = [d for d in data if d.datetime.date() < FIRE_DATE]
after  = [d for d in data if d.datetime.date() >= FIRE_DATE]

print(f"\nPre-fire passes  : {len(before)}")
print(f"Post-fire passes : {len(after)}")

if not before:
    print("WARNING: no pre-fire passes found.")
if not after:
    print("WARNING: no post-fire passes found.")


# ════════════════════════════════════════════════════════════════════════════
# 2. RTC composites — before vs after
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Colour composites — before / after fire")
print("=" * 60)

# Pick the last pre-fire pass and the first post-fire pass.
pre  = before[-1] if before else None
post = after[0]   if after  else None

items_to_show = [(lbl, it) for lbl, it in [("Pre-fire", pre), ("Post-fire", post)] if it is not None]

if len(items_to_show) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (label, item) in zip(axes, items_to_show):
        item.load()
        vv = item.data["VV"].values
        vh = item.data["VH"].values
        rgb = rtc_composite(vv, vh)
        ax.imshow(rgb, origin="upper")
        ax.set_title(f"{label}: {item.label}", fontsize=12)
        ax.set_axis_off()
        item.unload()
    fig.suptitle("Artillerie Schietkamp — 't Harde wildfire (29 Apr 2026)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fire_composite_sidebyside.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 3. All passes — composite overview
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. All passes — composite overview")
print("=" * 60)

ncols = min(len(data), 4)
nrows = int(np.ceil(len(data) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
axes = np.atleast_2d(axes)
for idx, d in enumerate(data):
    ax = axes.flat[idx]
    d.load()
    rgb = rtc_composite(d.data["VV"].values, d.data["VH"].values)
    ax.imshow(rgb, origin="upper")
    marker = " *FIRE*" if d.datetime.date() >= FIRE_DATE else ""
    ax.set_title(f"{d.label}{marker}", fontsize=9)
    ax.set_axis_off()
    d.unload()
# hide unused axes
for idx in range(len(data), nrows * ncols):
    axes.flat[idx].set_axis_off()

fig.suptitle("OPERA RTC-S1 — 't Harde (all passes)", fontsize=14)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fire_composite_allpasses.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Change map — log-ratio VV and VH
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Change map — log-ratio VV / VH")
print("=" * 60)

if pre is not None and post is not None:
    pre.load()
    post.load()

    vv_pre  = pre.data["VV"].values.astype(np.float32)
    vh_pre  = pre.data["VH"].values.astype(np.float32)
    vv_post = post.data["VV"].values.astype(np.float32)
    vh_post = post.data["VH"].values.astype(np.float32)

    pre.unload()
    post.unload()

    # Log-ratio change  (dB difference)
    eps = 1e-10
    vv_change_db = 10 * np.log10((vv_post + eps) / (vv_pre + eps))
    vh_change_db = 10 * np.log10((vh_post + eps) / (vh_pre + eps))

    # Mask out nodata
    mask = np.isnan(vv_pre) | np.isnan(vv_post) | np.isnan(vh_pre) | np.isnan(vh_post)
    vv_change_db[mask] = np.nan
    vh_change_db[mask] = np.nan

    # Combined change magnitude
    change_mag = np.sqrt(vv_change_db**2 + vh_change_db**2)
    change_mag[mask] = np.nan

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # VV change
    vmin, vmax = -5, 5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im0 = axes[0].imshow(vv_change_db, cmap="RdBu_r", norm=norm, origin="upper")
    axes[0].set_title(f"ΔVV (dB)\n{pre.label} → {post.label}", fontsize=11)
    axes[0].set_axis_off()
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="dB")

    # VH change
    im1 = axes[1].imshow(vh_change_db, cmap="RdBu_r", norm=norm, origin="upper")
    axes[1].set_title(f"ΔVH (dB)\n{pre.label} → {post.label}", fontsize=11)
    axes[1].set_axis_off()
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="dB")

    # Combined change magnitude
    im2 = axes[2].imshow(change_mag, cmap="hot_r", vmin=0, vmax=5, origin="upper")
    axes[2].set_title("Change magnitude (dB)", fontsize=11)
    axes[2].set_axis_off()
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="dB")

    fig.suptitle("RTC change detection — 't Harde wildfire", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fire_change_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Difference composite overlay ---
    # RGB: R=ΔVV, G=ΔVH, B=0  (red = VV increase, green = VH increase)
    dvv_norm = np.clip((vv_change_db + 5) / 10, 0, 1)
    dvh_norm = np.clip((vh_change_db + 5) / 10, 0, 1)
    diff_rgb = np.dstack([dvv_norm, dvh_norm, np.zeros_like(dvv_norm)])
    diff_rgb[mask] = 1.0  # white nodata

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(diff_rgb, origin="upper")
    ax.set_title(
        f"Change composite (R=ΔVV, G=ΔVH)\n{pre.label} → {post.label}",
        fontsize=12,
    )
    ax.set_axis_off()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fire_change_composite.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

else:
    print("Skipping change map — need both pre- and post-fire passes.")


print("\n" + "=" * 60)
print("Done.  All outputs in:", OUT_DIR)
print("=" * 60)
