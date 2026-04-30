"""
Sentinel-1 SLC-Burst oil slick analysis — Deurganckdock, Antwerp.

An oil slick formed in the Deurganckdock on the night of 9–10 April 2026.
This script:
  1. Searches ASF for Sentinel-1 SLC-BURST data over the dock area.
  2. Finds the closest **same-orbit** before/after pair (identical radar
     geometry for valid pixel-level comparison).
  3. Downloads the VV bursts and computes amplitude from the complex SLC.
  4. Produces a before/after slider comparison, side-by-side, and
     difference map.

**Polarization choice** — VV (co-pol) is used because oil dampens
short-gravity / capillary ocean waves, reducing Bragg scattering.
The VV channel has better signal-to-noise ratio than VH for detecting
this effect (Brekke & Solberg, 2005; Bianchi et al., 2020).

Usage:
    conda activate RS-applications
    python run_deurganckdock_oil_slick.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── output directory ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
GIF_DIR = os.path.join(OUT_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "bursts"), exist_ok=True)

# ── area of interest ─────────────────────────────────────────────────────────
# Deurganckdock, Antwerp port
BBOX = [4.251283836960951, 51.28632393164611,
        4.271149638243543, 51.298511604339325]

# ── event timing ──────────────────────────────────────────────────────────────
EVENT_DATE = "2026-04-10"        # Oil slick first observed
SEARCH_START = "2026-03-25"      # ~2 weeks before → catch at least 1 before
SEARCH_END   = "2026-04-25"      # ~2 weeks after  → catch at least 1 after


# ════════════════════════════════════════════════════════════════════════════
# 1. Search ASF for SLC-BURST data
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Searching ASF for Sentinel-1 SLC-BURST data …")
print("=" * 60)

import asf_search

wkt = (
    f"POLYGON(({BBOX[0]} {BBOX[1]},{BBOX[2]} {BBOX[1]},"
    f"{BBOX[2]} {BBOX[3]},{BBOX[0]} {BBOX[3]},{BBOX[0]} {BBOX[1]}))"
)

results = asf_search.search(
    dataset="SLC-BURST",
    intersectsWith=wkt,
    start=f"{SEARCH_START}T00:00:00Z",
    end=f"{SEARCH_END}T23:59:59Z",
    maxResults=250,
)

print(f"\n→ ASF returned {len(results)} burst scenes")
if not results:
    print("No results found. Exiting.")
    sys.exit(0)


# ════════════════════════════════════════════════════════════════════════════
# 2. Group by burst ID → find same-orbit before/after pair
# ════════════════════════════════════════════════════════════════════════════
from datetime import datetime
from collections import defaultdict

# Group VV scenes by burst key (burst_id + swath → same radar geometry)
burst_groups = defaultdict(list)
for r in results:
    props = r.geojson().get("properties", {})
    pol = props.get("polarization", "")
    scene = props.get("sceneName", "")
    start = props.get("startTime", "")
    fdir = props.get("flightDirection", "")
    if "VV" not in pol or not start:
        continue
    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    parts = scene.replace("-BURST", "").split("_")
    burst_key = parts[1] + "_" + parts[2]   # e.g. "343971_IW2"
    burst_groups[burst_key].append((dt, scene, fdir, r))

event_dt = datetime.fromisoformat(f"{EVENT_DATE}T00:00:00+00:00")

# Find the best same-orbit pair that brackets the event
best_pair = None
best_gap = float("inf")

for key, entries in burst_groups.items():
    dates_sorted = sorted(entries, key=lambda x: x[0])
    before = [e for e in dates_sorted if e[0] < event_dt]
    after  = [e for e in dates_sorted if e[0] >= event_dt]
    if before and after:
        gap = (after[0][0] - before[-1][0]).days
        if gap < best_gap:
            best_gap = gap
            best_pair = (key, before[-1], after[0])

if best_pair is None:
    print("\n⚠  Could not find a same-orbit before/after pair.")
    sys.exit(1)

burst_key, before_entry, after_entry = best_pair
before_dt, before_scene, before_dir, before_obj = before_entry
after_dt, after_scene, after_dir, after_obj = after_entry
before_key = before_dt.strftime("%Y-%m-%d")
after_key = after_dt.strftime("%Y-%m-%d")

print(f"\n→ Best same-orbit pair: {burst_key} ({before_dir})")
print(f"  Before: {before_key}  {before_scene}")
print(f"  After:  {after_key}  {after_scene}")
print(f"  Gap:    {best_gap} days")


# ════════════════════════════════════════════════════════════════════════════
# 3. Download VV bursts (before & after, same orbit)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Downloading VV SLC bursts …")
print("=" * 60)

from pathlib import Path
import netrc as _netrc

netrc_path = Path.home() / ".netrc"
info = _netrc.netrc(str(netrc_path))
auth = info.authenticators("urs.earthdata.nasa.gov")
if not auth:
    print("ERROR: No credentials for urs.earthdata.nasa.gov in ~/.netrc")
    sys.exit(1)

session = asf_search.ASFSession().auth_with_creds(auth[0], auth[2])


def download_burst(scene_obj, date_key: str, label: str) -> str:
    """Download a single burst TIFF, return its local path."""
    dl_dir = os.path.join(OUT_DIR, "bursts", f"{label}_{date_key}")
    os.makedirs(dl_dir, exist_ok=True)

    existing = [os.path.join(dl_dir, f)
                for f in os.listdir(dl_dir) if f.endswith((".tif", ".tiff"))]
    if existing:
        print(f"  {label}: already on disk → {existing[0]}")
        return existing[0]

    print(f"  {label}: downloading …")
    scene_obj.download(dl_dir, session=session)
    downloaded = [os.path.join(dl_dir, f)
                  for f in os.listdir(dl_dir) if f.endswith((".tif", ".tiff"))]
    print(f"  {label}: → {downloaded[0]}")
    return downloaded[0]


before_file = download_burst(before_obj, before_key, "before_same_orbit")
after_file  = download_burst(after_obj, after_key, "after")


# ════════════════════════════════════════════════════════════════════════════
# 4. Load SLC data and compute amplitude
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Computing SLC amplitude …")
print("=" * 60)

import rasterio


def slc_to_intensity_db(tif_path: str) -> np.ndarray:
    """Read a complex SLC GeoTIFF and return intensity in dB."""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
    if np.iscomplexobj(data):
        amp = np.abs(data).astype(np.float32)
    else:
        amp = data.astype(np.float32)
    intensity = amp ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(intensity)
    db[~np.isfinite(db)] = np.nan
    return db


before_db = slc_to_intensity_db(before_file)
after_db  = slc_to_intensity_db(after_file)

print(f"  Before shape: {before_db.shape}")
print(f"  After  shape: {after_db.shape}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Before/after slider comparison
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Creating before/after comparison …")
print("=" * 60)

from rs_tools.visualization.slider import slider_comparison

# Crop to the same dimensions (same orbit → nearly identical, may differ by 1-2 rows)
min_rows = min(before_db.shape[0], after_db.shape[0])
min_cols = min(before_db.shape[1], after_db.shape[1])
before_crop = before_db[:min_rows, :min_cols]
after_crop  = after_db[:min_rows, :min_cols]

# Compute common dB range for consistent display
valid_before = before_crop[np.isfinite(before_crop)]
valid_after  = after_crop[np.isfinite(after_crop)]
all_valid = np.concatenate([valid_before, valid_after])
vmin = float(np.nanpercentile(all_valid, 2))
vmax = float(np.nanpercentile(all_valid, 98))

fig = slider_comparison(
    before_crop,
    after_crop,
    left_label=f"Before ({before_key})",
    right_label=f"After ({after_key})",
    cmap="gray",
    vmin=vmin,
    vmax=vmax,
    title=f"Deurganckdock Oil Slick — VV Backscatter (dB)\n"
          f"Before: {before_key}  |  After: {after_key}  "
          f"(same orbit: {burst_key})",
    figsize=(14, 8),
)

slider_path = os.path.join(OUT_DIR, "oil_slick_slider.png")
fig.savefig(slider_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {slider_path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. Static side-by-side comparison
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].imshow(before_crop, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
axes[0].set_title(f"Before — {before_key} (same orbit)", fontsize=12)
axes[0].set_axis_off()

axes[1].imshow(after_crop, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
axes[1].set_title(f"After — {after_key} (same orbit)", fontsize=12)
axes[1].set_axis_off()

fig.suptitle(
    f"Deurganckdock Oil Slick — Sentinel-1 VV SLC Burst (dB)\n"
    f"Same orbit track ({burst_key}, {before_dir.lower()}) "
    f"— Oil dampens surface waves → dark patch on water",
    fontsize=13,
)
plt.tight_layout()
sideby_path = os.path.join(OUT_DIR, "oil_slick_sidebyside.png")
fig.savefig(sideby_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {sideby_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. Difference map (after − before)
# ════════════════════════════════════════════════════════════════════════════
diff = after_crop - before_crop

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(diff, cmap="RdBu_r", vmin=-5, vmax=5, origin="upper")
ax.set_title(
    f"Backscatter difference (dB):  {after_key} − {before_key}\n"
    "Blue = darker after (potential oil slick)  |  Red = brighter after",
    fontsize=12,
)
ax.set_axis_off()
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔdB")
plt.tight_layout()
diff_path = os.path.join(OUT_DIR, "oil_slick_difference.png")
fig.savefig(diff_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {diff_path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. Analysis: can we see the oil slick?
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Quick analysis")
print("=" * 60)

mean_before = float(np.nanmean(valid_before))
mean_after  = float(np.nanmean(valid_after))
delta_db    = mean_after - mean_before

valid_diff = diff[np.isfinite(diff)]
pct_darkened = float(np.sum(valid_diff < -3.0)) / max(len(valid_diff), 1) * 100

print(f"  Mean VV backscatter before:  {mean_before:.2f} dB")
print(f"  Mean VV backscatter after:   {mean_after:.2f} dB")
print(f"  Difference:                  {delta_db:+.2f} dB")
print(f"  Pixels darkened > 3 dB:      {pct_darkened:.1f}%")

if delta_db < -1.0:
    print("\n  → Significant overall darkening detected.")
    print("    This is consistent with oil dampening Bragg wave scattering,")
    print("    though part of the signal may also reflect changing sea state / wind.")
elif pct_darkened > 5:
    print("\n  → Localised darkening detected — oil slick likely confined to part of the burst.")
elif delta_db < 0:
    print("\n  → Slight darkening — possible oil signal mixed with other effects (wind, tide).")
else:
    print("\n  → No clear darkening — oil slick may be localised or below detection threshold.")

print("\n  Note: The burst covers a full IW swath (~250 km). The Deurganckdock")
print("  is a small fraction of this area. For precise slick delineation,")
print("  geocoding the SLC data would allow cropping to the dock footprint.")

print("\nDone. All outputs written to:", OUT_DIR)
