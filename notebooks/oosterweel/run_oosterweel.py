"""
OPERA RTC-S1 time-series analysis — Oosterweel construction works, Antwerp.

Runs the full pipeline (search → load → composite → GIF → backscatter plot)
without Jupyter.  Uses the Terrascope archive.

Usage:
    conda activate RS-applications
    python run_oosterweel.py
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

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk, LoadedItem
from rs_tools.visualization.rtc_composite import rtc_composite
from rs_tools.visualization.scalebar import add_scalebar
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.overlays import (
    fetch_roads, overlay_roads, annotate_location, overlay_geojson,
)

# ── output directory ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
GIF_DIR = os.path.join(OUT_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "passes"), exist_ok=True)

# ── area of interest ─────────────────────────────────────────────────────────
bbox_oosterweel = BoundingBox(west=4.30, south=51.17, east=4.48, north=51.27)

# ── construction trajectory GeoJSON ─────────────────────────────────────────
GEOJSON_PATH = os.path.join(OUT_DIR, "oosterweel_trajectory.geojson")

# Categories to overlay (skip Oosterweelsteenweg to avoid clutter)
TRAJECTORY_CATEGORIES = [
    "Scheldetunnel", "Kanaaltunnels", "Bypass R1",
    "Motorway R1 (under construction)",
    "Oosterweelknooppunt (construction zone)",
    "Scheldetunnel construction site",
    "Junction construction site",
]
TRAJECTORY_STYLES = {
    "Scheldetunnel":                     {"color": "lime",   "linewidth": 2.5, "linestyle": "--"},
    "Kanaaltunnels":                     {"color": "cyan",   "linewidth": 2.5, "linestyle": "--"},
    "Bypass R1":                         {"color": "magenta","linewidth": 2.0},
    "Motorway R1 (under construction)":  {"color": "red",    "linewidth": 1.8},
    "Oosterweelknooppunt (construction zone)": {"color": "orange", "linewidth": 1.5, "alpha": 0.5},
    "Scheldetunnel construction site":   {"color": "lime",   "linewidth": 1.2, "alpha": 0.4},
    "Junction construction site":        {"color": "orange", "linewidth": 1.2, "alpha": 0.4},
}

# ── temporal range & sampling ────────────────────────────────────────────────
START_DATE      = "2014-10-01"
END_DATE        = "2026-03-24"
INTERVAL_MONTHS = 1           # monthly sampling


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading Oosterweel data …")
print("=" * 60)
data = load_dataset(
    "OPERA_RTC_S1",
    bbox=bbox_oosterweel,
    start_date=START_DATE,
    end_date=END_DATE,
    archive="terrascope",
    limit=500,
    monthly=True,
    interval_months=INTERVAL_MONTHS,
    output_dir=OUT_DIR,
)
print(f"\n→ Saved {len(data)} passes to {OUT_DIR}/passes/")

# Reload metadata from disk (pixel data stays on disk)
data = load_passes_from_disk(OUT_DIR)
print(f"→ {len(data)} passes available on disk\n")


# ════════════════════════════════════════════════════════════════════════════
# 2. Inspect
# ════════════════════════════════════════════════════════════════════════════
print("Oosterweel items:")
for item in data:
    print(f"  {item.label}  CRS={item.crs}  pixel={item.pixel_size_m} m  on_disk={item.pass_dir}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Acquisition timeline
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 3))
dates = [item.datetime for item in data]
ax.scatter(dates, [0] * len(dates), marker="|", s=300,
           c="darkorange", linewidths=2, label=f"Oosterweel ({len(dates)})")
for item in data:
    ax.annotate(
        item.platform.split("-")[1] if "-" in item.platform else item.platform,
        (item.datetime, 0), fontsize=6, ha="center", va="top",
        xytext=(0, -4), textcoords="offset points", color="darkorange",
    )
ax.set_yticks([])
ax.set_title("OPERA RTC-S1 acquisition timeline — Oosterweel")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
timeline_path = os.path.join(OUT_DIR, "timeline.png")
fig.savefig(timeline_path, dpi=150)
plt.close(fig)
print(f"\nSaved: {timeline_path}")


# ════════════════════════════════════════════════════════════════════════════
# 4. RTC composite — latest scene with road overlay & annotations
# ════════════════════════════════════════════════════════════════════════════
if data:
    item = data[-1]
    item.load()
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    rgb = rtc_composite(vv, vh)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    ax.set_title(f"Oosterweel — {item.label}", fontsize=12)

    # Road overlay
    try:
        roads = fetch_roads(
            bbox=(bbox_oosterweel.south, bbox_oosterweel.west,
                  bbox_oosterweel.north, bbox_oosterweel.east),
            highway_types="motorway|trunk|primary|secondary",
        )
        overlay_roads(ax, roads, item.data["VV"],
                      color="#888888", linewidth=0.4, alpha=0.5)
    except Exception as e:
        print(f"  (road overlay skipped: {e})")

    # Construction trajectory overlay
    if os.path.exists(GEOJSON_PATH):
        overlay_geojson(
            ax, GEOJSON_PATH, item.data["VV"],
            category_styles=TRAJECTORY_STYLES,
            filter_categories=TRAJECTORY_CATEGORIES,
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.7,
                  facecolor="black", edgecolor="gray", labelcolor="white")
    else:
        print(f"  ⚠ GeoJSON not found: {GEOJSON_PATH}")
        print(f"    Run: python fetch_oosterweel_geojson.py --output-dir {OUT_DIR}")

    # Annotate key locations
    annotations = [
        (4.3925, 51.2340, "Oosterweel\ntunnel north"),
        (4.3885, 51.2210, "Scheldt\ncrossing"),
        (4.4035, 51.2140, "Linkeroever\ntunnel south"),
        (4.4220, 51.2230, "Port of\nAntwerp"),
        (4.3560, 51.2170, "Antwerp\ncity centre"),
        (4.4150, 51.2470, "R1 ring\nmotorway"),
    ]
    for lon, lat, label in annotations:
        try:
            annotate_location(ax, lon, lat, label, item.data["VV"],
                              color="cyan", fontsize=7, markersize=3)
        except Exception:
            pass

    ax.text(0.02, 0.02, item.label, transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            verticalalignment="bottom")
    if item.pixel_size_m:
        add_scalebar(ax, item.pixel_size_m)
    plt.tight_layout()
    ow_path = os.path.join(OUT_DIR, "oosterweel_latest.png")
    fig.savefig(ow_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ow_path}")
    item.unload()


# ════════════════════════════════════════════════════════════════════════════
# 5. Animated GIF
# ════════════════════════════════════════════════════════════════════════════
def _rtc_composite_fn(item: LoadedItem):
    item.load()
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    rgb = rtc_composite(vv, vh)
    label = item.label
    item.unload()
    return rgb, label

if data:
    gif_path = os.path.join(GIF_DIR, "oosterweel_rtc.gif")
    save_timeseries_gif_lazy(
        data, gif_path,
        composite_fn=_rtc_composite_fn,
        title="Oosterweel — OPERA RTC-S1",
        pixel_size_m=data[0].pixel_size_m, fps=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. Backscatter time-series (dB) — point or AOI mean
# ════════════════════════════════════════════════════════════════════════════
POINT = (4.3925, 51.2340)   # Oosterweel tunnel north entrance (lon, lat)
# POINT = None              # → spatial mean over entire AOI

def backscatter_db(items, point=None, crs="EPSG:4326"):
    import pyproj
    dates, vv_db, vh_db, sensors = [], [], [], []
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
    return dates, vv_db, vh_db, sensors

if data:
    fig, ax = plt.subplots(figsize=(14, 5))
    dates, vv, vh, sensors = backscatter_db(data, point=POINT)
    ax.plot(dates, vv, "o-", color="steelblue", ms=5, lw=1.2, label="VV (dB)")
    ax.plot(dates, vh, "s-", color="darkorange", ms=5, lw=1.2, label="VH (dB)")
    for d, v, s in zip(dates, vv, sensors):
        ax.annotate(s, (d, v), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", color="steelblue")
    ax.set_ylabel("Backscatter (dB)")
    ax.set_xlabel("Acquisition date")
    if POINT:
        ax.set_title(f"Oosterweel — VV / VH backscatter at ({POINT[0]:.4f}, {POINT[1]:.4f})")
    else:
        ax.set_title("Oosterweel — VV / VH backscatter (AOI mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    bs_path = os.path.join(OUT_DIR, "backscatter_timeseries.png")
    fig.savefig(bs_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bs_path}")

print("\nDone. All outputs written to:", OUT_DIR)
