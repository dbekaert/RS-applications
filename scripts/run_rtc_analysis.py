"""
Standalone OPERA RTC-S1 time-series analysis.

Runs the full pipeline (search → load → composite → GIF → backscatter plot)
without Jupyter.  Uses the Terrascope archive.

Usage:
    python scripts/run_rtc_analysis.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Make sure the package is importable whether installed or not
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed; saves to files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, LoadedItem
from rs_tools.visualization.rtc_composite import rtc_composite
from rs_tools.visualization.scalebar import add_scalebar
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.overlays import fetch_roads, overlay_roads, annotate_location

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.join(_repo, "output")
GIF_DIR = os.path.join(OUT_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)

# ── areas of interest ────────────────────────────────────────────────────────
bbox_windfarms  = BoundingBox(west=2.37, south=51.33, east=3.43, north=51.89)
bbox_oosterweel = BoundingBox(west=4.30, south=51.17, east=4.48, north=51.27)

# ── temporal range & sampling ────────────────────────────────────────────────
START_DATE      = "2024-01-01"
END_DATE        = "2026-03-22"
INTERVAL_MONTHS = 6          # one sample every 6 months → ~4 dates
RUN_WINDFARMS   = True
RUN_OOSTERWEEL  = False      # set True to also run Oosterweel


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════
data_wf, data_ow = [], []

if RUN_WINDFARMS:
    print("=" * 60)
    print("Loading Wind Farms data …")
    print("=" * 60)
    data_wf = load_dataset(
        "OPERA_RTC_S1",
        bbox=bbox_windfarms,
        start_date=START_DATE,
        end_date=END_DATE,
        archive="terrascope",
        limit=500,
        monthly=True,
        interval_months=INTERVAL_MONTHS,
    )
    print(f"\n→ Loaded {len(data_wf)} samples for Wind Farms\n")

if RUN_OOSTERWEEL:
    print("=" * 60)
    print("Loading Oosterweel data …")
    print("=" * 60)
    data_ow = load_dataset(
        "OPERA_RTC_S1",
        bbox=bbox_oosterweel,
        start_date=START_DATE,
        end_date=END_DATE,
        archive="terrascope",
        limit=500,
        monthly=True,
        interval_months=INTERVAL_MONTHS,
    )
    print(f"\n→ Loaded {len(data_ow)} samples for Oosterweel\n")


# ════════════════════════════════════════════════════════════════════════════
# 2. Inspect
# ════════════════════════════════════════════════════════════════════════════
print("Wind Farms items:")
for item in data_wf:
    vv = item.data.get("VV")
    shape = vv.shape if vv is not None else "N/A"
    print(f"  {item.label}  CRS={item.crs}  pixel={item.pixel_size_m} m  shape={shape}")

print("\nOosterweel items:")
for item in data_ow:
    print(f"  {item.label}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Acquisition timeline plot
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 3))
for items, y, color, name in [
    (data_wf, 1, "steelblue", "Wind farms"),
    (data_ow, 0, "darkorange", "Oosterweel"),
]:
    dates = [item.datetime for item in items]
    ax.scatter(dates, [y] * len(dates), marker="|", s=300,
               c=color, linewidths=2, label=f"{name} ({len(dates)})")
    for item in items:
        ax.annotate(
            item.platform.split("-")[1] if "-" in item.platform else item.platform,
            (item.datetime, y), fontsize=6, ha="center",
            va="bottom" if y == 1 else "top",
            xytext=(0, 4 if y == 1 else -4),
            textcoords="offset points", color=color,
        )
ax.set_yticks([])
ax.set_title("OPERA RTC-S1 acquisition timeline")
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
# 4. RTC composites — latest scene for each AOI
# ════════════════════════════════════════════════════════════════════════════
def save_composite(item: LoadedItem, out_path: str, title: str) -> None:
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    rgb = rtc_composite(vv, vh)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    ax.set_title(f"{title} — {item.label}", fontsize=12)
    ax.text(0.02, 0.02, item.label, transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            verticalalignment="bottom")
    if item.pixel_size_m:
        add_scalebar(ax, item.pixel_size_m)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if data_wf:
    save_composite(data_wf[-1],
                   os.path.join(OUT_DIR, "windfarms_latest.png"),
                   "North Sea Wind Farms")

if data_ow:
    item = data_ow[-1]
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


# ════════════════════════════════════════════════════════════════════════════
# 5. Animated GIFs
# ════════════════════════════════════════════════════════════════════════════
def _rtc_composite_fn(item: LoadedItem):
    vv = item.data["VV"].values
    vh = item.data["VH"].values
    return rtc_composite(vv, vh), item.label


if data_wf:
    gif_path = os.path.join(GIF_DIR, "windfarms_rtc.gif")
    save_timeseries_gif_lazy(
        data_wf, gif_path,
        composite_fn=_rtc_composite_fn,
        title="North Sea Wind Farms — OPERA RTC-S1",
        pixel_size_m=data_wf[0].pixel_size_m, fps=2,
    )

if data_ow:
    gif_path = os.path.join(GIF_DIR, "oosterweel_rtc.gif")
    save_timeseries_gif_lazy(
        data_ow, gif_path,
        composite_fn=_rtc_composite_fn,
        title="Oosterweel — OPERA RTC-S1",
        pixel_size_m=data_ow[0].pixel_size_m, fps=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. Backscatter time-series (dB)
# ════════════════════════════════════════════════════════════════════════════
def mean_backscatter_db(items):
    dates, vv_db, vh_db, sensors = [], [], [], []
    for item in items:
        vv = item.data["VV"].values
        vh = item.data["VH"].values
        vv_mean = np.nanmean(vv[vv > 0])
        vh_mean = np.nanmean(vh[vh > 0])
        if vv_mean > 0 and vh_mean > 0:
            dates.append(item.datetime)
            vv_db.append(10 * np.log10(vv_mean))
            vh_db.append(10 * np.log10(vh_mean))
            sensors.append(
                item.platform.split("-")[1] if "-" in item.platform else item.platform
            )
    return dates, vv_db, vh_db, sensors


if data_wf or data_ow:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, items, title in [
        (axes[0], data_wf,  "North Sea wind farms"),
        (axes[1], data_ow,  "Oosterweel"),
    ]:
        if not items:
            ax.set_title(f"{title} — no data")
            continue
        dates, vv, vh, sensors = mean_backscatter_db(items)
        ax.plot(dates, vv, "o-", color="steelblue", ms=5, lw=1.2, label="VV (dB)")
        ax.plot(dates, vh, "s-", color="darkorange", ms=5, lw=1.2, label="VH (dB)")
        for d, v, s in zip(dates, vv, sensors):
            ax.annotate(s, (d, v), fontsize=6, ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points", color="steelblue")
        ax.set_ylabel("Backscatter (dB)")
        ax.set_title(f"{title} — VV / VH backscatter")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Acquisition date")
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    plt.tight_layout()
    bs_path = os.path.join(OUT_DIR, "backscatter_timeseries.png")
    fig.savefig(bs_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {bs_path}")

print("\nDone. All outputs written to:", OUT_DIR)
