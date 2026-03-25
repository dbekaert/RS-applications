"""
Bypass R1 Oscillation Analysis — ASC vs DESC vs Combined.

Investigates the oscillation patterns visible on the Antwerp bypass (R1)
by separating ascending and descending passes.  SAR geometry differs
between orbit directions, which can cause apparent signal fluctuations
when passes are mixed.

Usage:
    conda activate RS-applications
    python run_bypass_analysis.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyproj

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk, LoadedItem
from rs_tools.visualization.rtc_composite import rtc_composite
from rs_tools.visualization.scalebar import add_scalebar
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.overlays import (
    fetch_roads, overlay_roads, annotate_location,
)

# ── output directory ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
GIF_DIR = os.path.join(OUT_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "passes"), exist_ok=True)

# ── area of interest: Antwerp bypass region (tighter around the R1 bypass) ─
# Focused on the bypass / Oosterweel junction area
bbox_bypass = BoundingBox(west=4.34, south=51.19, east=4.46, north=51.27)

# ── points of interest on the bypass for time-series extraction ─────────
BYPASS_POINTS = {
    "Bypass R1 north":     (4.4050, 51.2500),
    "Bypass R1 mid":       (4.4100, 51.2350),
    "Bypass R1 south":     (4.4050, 51.2200),
    "Scheldt crossing":    (4.3885, 51.2210),
    "Tunnel north":        (4.3925, 51.2340),
}

# ── temporal range & sampling ────────────────────────────────────────────────
START_DATE      = "2014-10-01"
END_DATE        = "2026-03-24"
INTERVAL_MONTHS = 1           # monthly for better oscillation characterisation


# ════════════════════════════════════════════════════════════════════════════
# Helper: extract backscatter time-series
# ════════════════════════════════════════════════════════════════════════════

def backscatter_db(items, point=None, crs="EPSG:4326"):
    """Extract backscatter in dB at a point or as AOI mean."""
    dates, vv_db, vh_db, sensors, orbits = [], [], [], [], []
    for item in items:
        item.load()
        vv_arr = item.data["VV"]
        vh_arr = item.data["VH"]

        if point is not None:
            raster_crs = vv_arr.rio.crs
            if raster_crs and str(raster_crs) != crs:
                transformer = pyproj.Transformer.from_crs(
                    crs, str(raster_crs), always_xy=True,
                )
                px, py = transformer.transform(point[0], point[1])
            else:
                px, py = point
            vv_val = float(vv_arr.sel(x=px, y=py, method="nearest").values)
            vh_val = float(vh_arr.sel(x=px, y=py, method="nearest").values)
        else:
            vv = vv_arr.values
            vh = vh_arr.values
            vv_val = float(np.nanmean(vv[vv > 0]))
            vh_val = float(np.nanmean(vh[vh > 0]))

        item.unload()

        if vv_val > 0 and vh_val > 0:
            dates.append(item.datetime)
            vv_db.append(10 * np.log10(vv_val))
            vh_db.append(10 * np.log10(vh_val))
            sensors.append(
                item.platform.split("-")[1] if "-" in item.platform else item.platform
            )
            orbits.append(item.orbit_direction or "unknown")
    return dates, vv_db, vh_db, sensors, orbits


def split_by_orbit(data):
    """Split passes into ascending, descending, and combined lists."""
    asc  = [d for d in data if (d.orbit_direction or "").lower().startswith("asc")]
    desc = [d for d in data if (d.orbit_direction or "").lower().startswith("desc")]
    return asc, desc, data


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading Antwerp bypass data …")
print("=" * 60)
data = load_dataset(
    "OPERA_RTC_S1",
    bbox=bbox_bypass,
    start_date=START_DATE,
    end_date=END_DATE,
    archive="terrascope",
    limit=500,
    monthly=True,
    interval_months=INTERVAL_MONTHS,
    output_dir=OUT_DIR,
)
print(f"\n→ Saved {len(data)} passes to {OUT_DIR}/passes/")

# Reload metadata from disk
data = load_passes_from_disk(OUT_DIR)
print(f"→ {len(data)} passes available on disk\n")

# Split by orbit direction
asc_data, desc_data, all_data = split_by_orbit(data)
print(f"  Ascending:  {len(asc_data)} passes")
print(f"  Descending: {len(desc_data)} passes")
print(f"  Total:      {len(all_data)} passes")


# ════════════════════════════════════════════════════════════════════════════
# 2. Acquisition timeline — colour-coded by orbit direction
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 3))
for item in asc_data:
    ax.scatter(item.datetime, 0.1, marker="|", s=300, c="steelblue", linewidths=2)
for item in desc_data:
    ax.scatter(item.datetime, -0.1, marker="|", s=300, c="darkorange", linewidths=2)
# Legend
ax.scatter([], [], marker="|", s=100, c="steelblue", linewidths=2, label=f"Ascending ({len(asc_data)})")
ax.scatter([], [], marker="|", s=100, c="darkorange", linewidths=2, label=f"Descending ({len(desc_data)})")
ax.set_yticks([0.1, -0.1])
ax.set_yticklabels(["ASC", "DESC"])
ax.set_ylim(-0.4, 0.4)
ax.set_title("OPERA RTC-S1 acquisition timeline — Antwerp bypass (by orbit direction)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
timeline_path = os.path.join(OUT_DIR, "bypass_timeline.png")
fig.savefig(timeline_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {timeline_path}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Backscatter time-series — ASC vs DESC vs combined at bypass points
# ════════════════════════════════════════════════════════════════════════════

for point_name, point_coords in BYPASS_POINTS.items():
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    datasets = [
        ("Ascending only",  asc_data,  "steelblue"),
        ("Descending only", desc_data, "darkorange"),
        ("Combined",        all_data,  "forestgreen"),
    ]

    for ax, (label, subset, color) in zip(axes, datasets):
        if not subset:
            ax.set_title(f"{label} — no data")
            continue
        dates, vv, vh, sensors, orbits = backscatter_db(subset, point=point_coords)
        ax.plot(dates, vv, "o-", color=color, ms=4, lw=1.0, alpha=0.8, label="VV (dB)")
        ax.plot(dates, vh, "s-", color=color, ms=4, lw=1.0, alpha=0.5, label="VH (dB)")
        ax.set_ylabel("Backscatter (dB)")
        ax.set_title(f"{label} — {point_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Acquisition date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.suptitle(f"Bypass oscillation analysis — {point_name} ({point_coords[0]:.4f}, {point_coords[1]:.4f})",
                 fontsize=13)
    fig.autofmt_xdate()
    plt.tight_layout()
    safe_name = point_name.replace(" ", "_").lower()
    ts_path = os.path.join(OUT_DIR, f"bypass_timeseries_{safe_name}.png")
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ts_path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. Standard deviation comparison — ASC vs DESC vs combined
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Oscillation strength (VV std dev in dB) at bypass points")
print("=" * 60)
print(f"{'Point':<25s}  {'ASC σ':>8s}  {'DESC σ':>8s}  {'Combined σ':>10s}")
print("-" * 60)

for point_name, point_coords in BYPASS_POINTS.items():
    stds = {}
    for label, subset in [("ASC", asc_data), ("DESC", desc_data), ("Combined", all_data)]:
        if not subset:
            stds[label] = float("nan")
            continue
        _, vv, _, _, _ = backscatter_db(subset, point=point_coords)
        stds[label] = np.std(vv) if vv else float("nan")
    print(f"{point_name:<25s}  {stds['ASC']:>8.3f}  {stds['DESC']:>8.3f}  {stds['Combined']:>10.3f}")


# ════════════════════════════════════════════════════════════════════════════
# 5. RTC composites — ASC vs DESC side by side (latest of each)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, subset, label, color in [
    (axes[0], asc_data,  "Ascending",  "steelblue"),
    (axes[1], desc_data, "Descending", "darkorange"),
]:
    if not subset:
        ax.set_title(f"{label} — no data")
        continue
    item = subset[-1]
    item.load()
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    rgb = rtc_composite(vv, vh)
    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    ax.set_title(f"{label} — {item.label}", fontsize=11)

    # Annotate bypass points
    for pname, pcoords in BYPASS_POINTS.items():
        try:
            annotate_location(ax, pcoords[0], pcoords[1], pname,
                              item.data["VV"], color="cyan", fontsize=7, markersize=3)
        except Exception:
            pass

    if item.pixel_size_m:
        add_scalebar(ax, item.pixel_size_m)
    item.unload()

plt.suptitle("Antwerp Bypass — Ascending vs Descending geometry", fontsize=13)
plt.tight_layout()
comp_path = os.path.join(OUT_DIR, "bypass_asc_vs_desc.png")
fig.savefig(comp_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {comp_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Animated GIFs — separate ASC, DESC, and combined
# ════════════════════════════════════════════════════════════════════════════

def _rtc_composite_fn(item):
    item.load()
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    rgb = rtc_composite(vv, vh)
    label = item.label
    item.unload()
    return rgb, label

for subset, name in [(asc_data, "ascending"), (desc_data, "descending"), (all_data, "combined")]:
    if not subset:
        print(f"Skipping {name} GIF — no data")
        continue
    gif_path = os.path.join(GIF_DIR, f"bypass_{name}.gif")
    save_timeseries_gif_lazy(
        subset, gif_path,
        composite_fn=_rtc_composite_fn,
        title=f"Antwerp Bypass — {name.title()}",
        pixel_size_m=subset[0].pixel_size_m if subset[0].pixel_size_m else None,
        fps=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# 7. Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  AOI: {bbox_bypass}")
print(f"  Ascending passes:  {len(asc_data)}")
print(f"  Descending passes: {len(desc_data)}")
print(f"  Total passes:      {len(all_data)}")
print(f"\nLook at the time-series plots to see if the oscillation is")
print(f"driven by mixing ASC/DESC geometries.  If the ASC-only and")
print(f"DESC-only series are smoother than the combined one, the")
print(f"oscillation is a viewing-geometry artefact.")
print(f"\nAll outputs written to: {OUT_DIR}")
