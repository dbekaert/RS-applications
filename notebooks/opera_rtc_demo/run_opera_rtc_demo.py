"""OPERA RTC-S1 Quick-Start Demo — standalone script.

Companion to the ``opera_rtc_demo.ipynb`` notebook.
Runs the same four demos non-interactively and saves all plots to disk.

Usage:
    conda activate RS-applications
    python run_opera_rtc_demo.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure rs_tools is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rs_tools.config import BoundingBox, SearchConfig
from rs_tools.search import search_archive
from rs_tools.datasets import (
    summarize_search_results,
    print_coverage_report,
    filter_by_coverage,
    records_to_items,
    load_items,
    load_dataset,
)
from rs_tools.datasets.mosaic import mosaic_items
from rs_tools.visualization.rtc_composite import rtc_composite, PRESETS
from rioxarray.merge import merge_arrays

# ── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

bbox = BoundingBox(west=4.25, south=51.20, east=4.45, north=51.35)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Search & Download
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Searching for OPERA RTC-S1 products (Terrascope)")
print("=" * 60)

config = SearchConfig(
    start_date="2024-06-01",
    end_date="2024-06-30",
    bbox=bbox,
    limit=200,
)

items = search_archive("terrascope", config)
print(f"Found {len(items)} STAC items")

records = summarize_search_results(items, bbox)
print_coverage_report(records)

selected = filter_by_coverage(
    records, min_coverage_pct=80.0, orbit_direction="ascending",
)
print(f"\nSelected {len(selected)} ascending passes with ≥ 80 % coverage")

pass_items = records_to_items(selected[:2])
data = load_items(pass_items, assets=["VV", "VH"], bbox=bbox, mosaic=True)

for d in data:
    print(f"  {d.label}  shape={d.data['VV'].shape}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Colour Composites
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Colour composites — adjusting limits")
print("=" * 60)

item = data[0]
vv, vh = item.data["VV"].values, item.data["VH"].values
print(f"Using: {item.label}  shape={vv.shape}")

# 2a — Preset comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, preset in zip(axes, ["default", "OPERA_global"]):
    rgb = rtc_composite(vv, vh, preset=preset)
    ax.imshow(rgb, origin="upper")
    ax.set_title(f'Preset: "{preset}"', fontsize=12)
    ax.set_axis_off()
fig.suptitle(f"Colour composite — {item.label}", fontsize=14)
plt.tight_layout()
path = os.path.join(OUT_DIR, "2a_preset_comparison.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# 2b — Custom ranges
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
configs = [
    ("Narrow (high contrast)", (0.20, 0.60), (0.06, 0.25)),
    ("Default",                *PRESETS["default"]),
    ("Wide (low saturation)",  (0.05, 1.2),  (0.02, 0.50)),
]
for ax, (label, co, cross) in zip(axes, configs):
    rgb = rtc_composite(vv, vh, co_pol_range=co, cross_pol_range=cross)
    ax.imshow(rgb, origin="upper")
    ax.set_title(f"{label}\nVV={co}, VH={cross}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Effect of colour limits on contrast", fontsize=14)
plt.tight_layout()
path = os.path.join(OUT_DIR, "2b_colour_limits.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Mosaic / Stitch Two Passes
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Mosaic — stitching two passes")
print("=" * 60)

# 3a — Individual passes
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, d in zip(axes, data[:2]):
    rgb = rtc_composite(d.data["VV"].values, d.data["VH"].values)
    ax.imshow(rgb, origin="upper")
    ax.set_title(d.label, fontsize=11)
    ax.set_axis_off()
fig.suptitle("Individual passes (before stitching)", fontsize=14)
plt.tight_layout()
path = os.path.join(OUT_DIR, "3a_individual_passes.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# 3b — Stitched
vv_stitched = merge_arrays([d.data["VV"] for d in data[:2]])
vh_stitched = merge_arrays([d.data["VH"] for d in data[:2]])
rgb_stitched = rtc_composite(vv_stitched.values, vh_stitched.values)

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(rgb_stitched, origin="upper")
ax.set_title("Stitched mosaic — two passes combined", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
path = os.path.join(OUT_DIR, "3b_stitched.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

del vv_stitched, vh_stitched, rgb_stitched


# ═══════════════════════════════════════════════════════════════════════════
# 4. Backscatter Conversion: gamma-0 → sigma-0 → beta-0
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Backscatter conversion: γ⁰ → σ⁰ / β⁰")
print("=" * 60)

common = dict(
    short_name="OPERA_RTC_S1",
    bbox=bbox,
    start_date="2024-06-01",
    end_date="2024-06-15",
    archive="terrascope",
    limit=20,
)

g0 = load_dataset(**common, backscatter="gamma0")
s0 = load_dataset(**common, backscatter="sigma0")
b0 = load_dataset(**common, backscatter="beta0")

for label, loaded in [("γ⁰", g0), ("σ⁰", s0), ("β⁰", b0)]:
    m = loaded[0]
    vv_mean = float(m.data["VV"].mean(skipna=True))
    vh_mean = float(m.data["VH"].mean(skipna=True))
    print(f"  {label}:  VV mean = {vv_mean:.4f},  VH mean = {vh_mean:.4f}")

# 4a — Side-by-side composites
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (label, loaded) in zip(axes, [
    ("γ⁰ (gamma-0)", g0),
    ("σ⁰ (sigma-0)", s0),
    ("β⁰ (beta-0)",  b0),
]):
    m = loaded[0]
    rgb = rtc_composite(m.data["VV"].values, m.data["VH"].values)
    ax.imshow(rgb, origin="upper")
    ax.set_title(label, fontsize=13)
    ax.set_axis_off()
fig.suptitle("Backscatter types — same pass", fontsize=14)
plt.tight_layout()
path = os.path.join(OUT_DIR, "4a_backscatter_types.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# 4b — VV histogram comparison
fig, ax = plt.subplots(figsize=(10, 5))
for label, loaded, colour in [
    ("γ⁰", g0, "steelblue"),
    ("σ⁰", s0, "darkorange"),
    ("β⁰", b0, "seagreen"),
]:
    vals = loaded[0].data["VV"].values.ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.hist(vals, bins=200, range=(0, 1.0), density=True, alpha=0.5,
            color=colour, label=f"{label} (mean = {vals.mean():.3f})")
ax.set_xlabel("Linear power", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("VV backscatter distribution — γ⁰ vs σ⁰ vs β⁰", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
path = os.path.join(OUT_DIR, "4b_backscatter_histograms.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Done — all plots saved to:", OUT_DIR)
print("=" * 60)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
