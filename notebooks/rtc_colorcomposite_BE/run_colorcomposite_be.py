"""
RTC Colour Composite Optimisation for Belgium.

Downloads OPERA RTC-S1 passes over Belgium (~every 2 months) and
analyses amplitude saturation with the current ASF/HyP3 colour ranges.
Proposes optimised ranges tuned for Belgian land cover (urban, farmland,
forests) and produces before / after comparisons plus an animated GIF.

Usage:
    conda activate RS-applications
    python run_colorcomposite_be.py
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
import rasterio
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.transform import from_bounds

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk, LoadedItem
from rs_tools.visualization.rtc_composite import rtc_composite, _CO_POL_RANGE, _CROSS_POL_RANGE
from rs_tools.visualization.scalebar import add_scalebar
from rs_tools.visualization.animation import save_timeseries_gif_lazy

# ── output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.expanduser("~/RS_applications/Applications/RTC/BelgiumColors")
GIF_DIR = os.path.join(OUT_DIR, "gifs")
os.makedirs(GIF_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "passes"), exist_ok=True)

# ── area of interest: Belgium ────────────────────────────────────────────────
# Covers the full territory of Belgium
bbox_belgium = BoundingBox(west=2.54, south=49.50, east=6.41, north=51.50)

# ── temporal range & sampling ────────────────────────────────────────────────
START_DATE      = "2014-10-01"
END_DATE        = "2026-03-24"
INTERVAL_MONTHS = 1           # one pass per month

# ── Default vs custom colour ranges ─────────────────────────────────────────
# ASF/HyP3 world-wide defaults (amplitude after sqrt)
DEFAULT_CO_POL   = _CO_POL_RANGE        # (0.14, 0.52)
DEFAULT_CROSS_POL = _CROSS_POL_RANGE    # (0.05, 0.259)


# ════════════════════════════════════════════════════════════════════════════
# Helper: analyse amplitude distribution
# ════════════════════════════════════════════════════════════════════════════

def analyse_amplitude_stats(data, tag="", max_pixels_per_pass=500_000):
    """Compute amplitude (sqrt of power) percentiles across all passes.

    To avoid OOM on large AOIs, a random subsample of pixels is taken
    from each pass (default 500k pixels — plenty for stable percentiles).

    Returns dict with VV and VH statistics.
    """
    rng = np.random.default_rng(42)
    vv_vals, vh_vals = [], []
    for item in data:
        item.load()
        vv = item.data["VV"].values.ravel()
        vh = item.data["VH"].values.ravel()
        item.unload()
        # Convert to amplitude
        mask_vv = np.isfinite(vv) & (vv > 0)
        mask_vh = np.isfinite(vh) & (vh > 0)
        vv_amp = np.sqrt(vv[mask_vv])
        vh_amp = np.sqrt(vh[mask_vh])
        # Subsample to keep memory bounded
        if len(vv_amp) > max_pixels_per_pass:
            idx = rng.choice(len(vv_amp), max_pixels_per_pass, replace=False)
            vv_amp = vv_amp[idx]
        if len(vh_amp) > max_pixels_per_pass:
            idx = rng.choice(len(vh_amp), max_pixels_per_pass, replace=False)
            vh_amp = vh_amp[idx]
        vv_vals.append(vv_amp)
        vh_vals.append(vh_amp)

    vv_all = np.concatenate(vv_vals)
    vh_all = np.concatenate(vh_vals)

    percentiles = [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99]
    stats = {
        "VV": {f"p{p}": float(np.percentile(vv_all, p)) for p in percentiles},
        "VH": {f"p{p}": float(np.percentile(vh_all, p)) for p in percentiles},
        "n_pixels": len(vv_all),
        "n_passes": len(data),
    }
    stats["VV"]["mean"] = float(np.mean(vv_all))
    stats["VV"]["std"]  = float(np.std(vv_all))
    stats["VH"]["mean"] = float(np.mean(vh_all))
    stats["VH"]["std"]  = float(np.std(vh_all))

    print(f"\n{'='*60}")
    print(f"Amplitude statistics{' — ' + tag if tag else ''}")
    print(f"{'='*60}")
    print(f"Passes analysed: {len(data)}")
    print(f"Total valid pixels: {len(vv_all):,}")
    print(f"\nVV amplitude (sqrt of power):")
    for p in percentiles:
        print(f"  P{p:>2d}: {stats['VV'][f'p{p}']:.4f}")
    print(f"  Mean: {stats['VV']['mean']:.4f}  Std: {stats['VV']['std']:.4f}")
    print(f"\nVH amplitude (sqrt of power):")
    for p in percentiles:
        print(f"  P{p:>2d}: {stats['VH'][f'p{p}']:.4f}")
    print(f"  Mean: {stats['VH']['mean']:.4f}  Std: {stats['VH']['std']:.4f}")

    return stats, vv_all, vh_all


def suggest_ranges(stats):
    """Suggest Belgium-optimised colour ranges from percentile stats.

    Uses P2–P98 to avoid extreme outliers while reducing saturation.
    """
    co_min   = stats["VV"]["p2"]
    co_max   = stats["VV"]["p98"]
    cross_min = stats["VH"]["p2"]
    cross_max = stats["VH"]["p98"]
    return (co_min, co_max), (cross_min, cross_max)


def compute_saturation(data, co_range, cross_range, label=""):
    """Report fraction of pixels saturated (clipped at 0 or 1)."""
    sat_low, sat_high, total = 0, 0, 0
    for item in data:
        item.load()
        vv = item.data["VV"].values
        vh = item.data["VH"].values
        item.unload()

        vv_amp = np.sqrt(np.clip(vv[np.isfinite(vv) & (vv > 0)], 0, None))
        vh_amp = np.sqrt(np.clip(vh[np.isfinite(vh) & (vh > 0)], 0, None))

        # Saturation = fraction outside [min, max]
        vv_lo = np.sum(vv_amp < co_range[0])
        vv_hi = np.sum(vv_amp > co_range[1])
        vh_lo = np.sum(vh_amp < cross_range[0])
        vh_hi = np.sum(vh_amp > cross_range[1])

        sat_low  += vv_lo + vh_lo
        sat_high += vv_hi + vh_hi
        total    += len(vv_amp) + len(vh_amp)

    pct_lo  = 100.0 * sat_low / total if total else 0
    pct_hi  = 100.0 * sat_high / total if total else 0
    pct_tot = pct_lo + pct_hi
    print(f"\nSaturation{' — ' + label if label else ''}:")
    print(f"  Co-pol range:    {co_range}")
    print(f"  Cross-pol range: {cross_range}")
    print(f"  Saturated low:  {pct_lo:.2f}%")
    print(f"  Saturated high: {pct_hi:.2f}%")
    print(f"  Total saturated: {pct_tot:.2f}%")
    return pct_lo, pct_hi


def plot_amplitude_histograms(vv_all, vh_all, default_co, default_cross,
                              suggested_co, suggested_cross, out_path):
    """Plot amplitude histograms with default and suggested ranges."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # VV
    ax = axes[0]
    ax.hist(vv_all, bins=500, range=(0, 1.0), density=True, alpha=0.7,
            color="steelblue", label="VV amplitude")
    ax.axvline(default_co[0], color="red", ls="--", lw=1.5, label=f"Default min={default_co[0]:.3f}")
    ax.axvline(default_co[1], color="red", ls="-",  lw=1.5, label=f"Default max={default_co[1]:.3f}")
    ax.axvline(suggested_co[0], color="lime", ls="--", lw=1.5, label=f"Belgium min={suggested_co[0]:.3f}")
    ax.axvline(suggested_co[1], color="lime", ls="-",  lw=1.5, label=f"Belgium max={suggested_co[1]:.3f}")
    ax.set_title("VV (co-pol) amplitude distribution — Belgium", fontsize=12)
    ax.set_xlabel("Amplitude (sqrt of power)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)

    # VH
    ax = axes[1]
    ax.hist(vh_all, bins=500, range=(0, 0.6), density=True, alpha=0.7,
            color="darkorange", label="VH amplitude")
    ax.axvline(default_cross[0], color="red", ls="--", lw=1.5, label=f"Default min={default_cross[0]:.3f}")
    ax.axvline(default_cross[1], color="red", ls="-",  lw=1.5, label=f"Default max={default_cross[1]:.3f}")
    ax.axvline(suggested_cross[0], color="lime", ls="--", lw=1.5, label=f"Belgium min={suggested_cross[0]:.3f}")
    ax.axvline(suggested_cross[1], color="lime", ls="-",  lw=1.5, label=f"Belgium max={suggested_cross[1]:.3f}")
    ax.set_title("VH (cross-pol) amplitude distribution — Belgium", fontsize=12)
    ax.set_xlabel("Amplitude (sqrt of power)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Helper: composite with custom ranges
# ════════════════════════════════════════════════════════════════════════════

# Maximum pixel dimension (height or width) for visualisation.
_VIS_MAX_DIM = 2000


def _build_reference_grid(data, max_dim=_VIS_MAX_DIM):
    """Build an xarray DataArray covering the union extent of all passes.

    The reference is downsampled so the longest dimension <= *max_dim*
    pixels.  Every pass can then be ``reproject_match``-ed to this grid
    so all frames have identical shape and alignment.
    """
    lefts, bottoms, rights, tops = [], [], [], []
    crs = None
    native_res = None
    for item in data:
        tif = os.path.join(item.pass_dir, "VV.tif")
        with rasterio.open(tif) as src:
            b = src.bounds
            lefts.append(b.left); bottoms.append(b.bottom)
            rights.append(b.right); tops.append(b.top)
            if crs is None:
                crs = src.crs
                native_res = src.res[0]

    left, bottom = min(lefts), min(bottoms)
    right, top = max(rights), max(tops)

    native_w = int(round((right - left) / native_res))
    native_h = int(round((top - bottom) / native_res))
    factor = max(1, max(native_h, native_w) // max_dim)
    res = native_res * factor

    w = int(round((right - left) / res))
    h = int(round((top - bottom) / res))

    transform = from_bounds(left, bottom, right, top, w, h)

    y_coords = np.linspace(top - res / 2, bottom + res / 2, h)
    x_coords = np.linspace(left + res / 2, right - res / 2, w)

    ref = xr.DataArray(
        np.zeros((h, w), dtype=np.float32),
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )
    ref = ref.rio.write_crs(crs)
    ref = ref.rio.write_transform(transform)
    print(f"Reference grid: {w}x{h} px  (factor={factor}x, res={res:.0f} m)")
    return ref


def _load_on_grid(item, ref_grid):
    """Load VV/VH from *item*, reproject onto *ref_grid*, return arrays."""
    item.load()
    vv = item.data["VV"].rio.reproject_match(ref_grid).values
    vh = item.data["VH"].rio.reproject_match(ref_grid).values
    item.unload()
    return vv, vh


# ════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Loading Belgium-wide RTC data …")
print("=" * 60)
data = load_dataset(
    "OPERA_RTC_S1",
    bbox=bbox_belgium,
    start_date=START_DATE,
    end_date=END_DATE,
    archive="terrascope",
    limit=2000,
    monthly=True,
    interval_months=INTERVAL_MONTHS,
    output_dir=OUT_DIR,
)
print(f"\n→ Saved {len(data)} passes to {OUT_DIR}/passes/")

# Reload metadata from disk (pixel data stays on disk)
data = load_passes_from_disk(OUT_DIR)
print(f"→ {len(data)} passes available on disk")

# Build a fixed reference grid (union of all pass extents, downsampled)
ref_grid = _build_reference_grid(data)
print()


# ════════════════════════════════════════════════════════════════════════════
# 2. Inspect
# ════════════════════════════════════════════════════════════════════════════
print("Belgium items:")
for item in data:
    print(f"  {item.label}  on_disk={item.pass_dir}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Acquisition timeline
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 3))
dates_plot = [item.datetime for item in data]
ax.scatter(dates_plot, [0] * len(dates_plot), marker="|", s=300,
           c="darkorange", linewidths=2, label=f"Belgium ({len(dates_plot)})")
for item in data:
    ax.annotate(
        item.platform.split("-")[1] if "-" in item.platform else item.platform,
        (item.datetime, 0), fontsize=6, ha="center", va="top",
        xytext=(0, -4), textcoords="offset points", color="darkorange",
    )
ax.set_yticks([])
ax.set_title("OPERA RTC-S1 acquisition timeline — Belgium colour composite")
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
# 4. Amplitude statistics & range suggestion
# ════════════════════════════════════════════════════════════════════════════
stats, vv_all, vh_all = analyse_amplitude_stats(data, tag="Belgium")

suggested_co, suggested_cross = suggest_ranges(stats)
print(f"\n{'='*60}")
print(f"Suggested Belgium colour ranges")
print(f"{'='*60}")
print(f"  Co-pol (VV):    {suggested_co}")
print(f"  Cross-pol (VH): {suggested_cross}")
print(f"\n  Default co-pol:    {DEFAULT_CO_POL}")
print(f"  Default cross-pol: {DEFAULT_CROSS_POL}")

# Saturation comparison
compute_saturation(data, DEFAULT_CO_POL, DEFAULT_CROSS_POL, label="ASF defaults")
compute_saturation(data, suggested_co, suggested_cross, label="Belgium suggested")


# ════════════════════════════════════════════════════════════════════════════
# 5. Amplitude histograms
# ════════════════════════════════════════════════════════════════════════════
hist_path = os.path.join(OUT_DIR, "amplitude_histograms.png")
plot_amplitude_histograms(vv_all, vh_all,
                          DEFAULT_CO_POL, DEFAULT_CROSS_POL,
                          suggested_co, suggested_cross,
                          hist_path)
# Free large arrays
del vv_all, vh_all


# ════════════════════════════════════════════════════════════════════════════
# 6. Before / After comparison for several passes
# ════════════════════════════════════════════════════════════════════════════
N_COMPARE = min(6, len(data))
compare_items = [data[i * len(data) // N_COMPARE] for i in range(N_COMPARE)]

fig, axes = plt.subplots(N_COMPARE, 2, figsize=(16, 6 * N_COMPARE))
if N_COMPARE == 1:
    axes = axes[np.newaxis, :]

for row, item in enumerate(compare_items):
    vv, vh = _load_on_grid(item, ref_grid)

    rgb_default = rtc_composite(vv, vh,
                                co_pol_range=DEFAULT_CO_POL,
                                cross_pol_range=DEFAULT_CROSS_POL)
    rgb_belgium = rtc_composite(vv, vh,
                                co_pol_range=suggested_co,
                                cross_pol_range=suggested_cross)
    del vv, vh

    axes[row, 0].imshow(rgb_default, origin="upper")
    axes[row, 0].set_axis_off()
    axes[row, 0].set_title(f"Default — {item.label}", fontsize=10)
    del rgb_default

    axes[row, 1].imshow(rgb_belgium, origin="upper")
    axes[row, 1].set_axis_off()
    axes[row, 1].set_title(f"Belgium — {item.label}", fontsize=10)
    del rgb_belgium

plt.suptitle("Before (ASF defaults) vs After (Belgium-optimised)", fontsize=14, y=1.01)
plt.tight_layout()
compare_path = os.path.join(OUT_DIR, "before_after_comparison.png")
fig.savefig(compare_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {compare_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. Animated GIF — before vs after (side by side per frame)
# ════════════════════════════════════════════════════════════════════════════

def _side_by_side_composite(item):
    """Render a single frame with default (left) and Belgium (right)."""
    vv, vh = _load_on_grid(item, ref_grid)
    rgb_default = rtc_composite(vv, vh,
                                co_pol_range=DEFAULT_CO_POL,
                                cross_pol_range=DEFAULT_CROSS_POL)
    rgb_belgium = rtc_composite(vv, vh,
                                co_pol_range=suggested_co,
                                cross_pol_range=suggested_cross)
    del vv, vh
    label = item.label
    # Create side-by-side with a thin separator
    sep = np.ones((rgb_default.shape[0], 4, 3), dtype=np.float32) * 0.3
    rgb = np.concatenate([rgb_default, sep, rgb_belgium], axis=1)
    del rgb_default, rgb_belgium
    return rgb, label

if data:
    gif_path = os.path.join(GIF_DIR, "belgium_before_after.gif")
    save_timeseries_gif_lazy(
        data, gif_path,
        composite_fn=_side_by_side_composite,
        title="Belgium RTC — Default (left) vs Optimised (right)",
        pixel_size_m=data[0].pixel_size_m if data[0].pixel_size_m else None,
        fps=2,
        figsize=(16, 8),
    )


# ════════════════════════════════════════════════════════════════════════════
# 8. GIF with Belgium-optimised colours only
# ════════════════════════════════════════════════════════════════════════════
def _optimised_composite(item):
    """Render a single frame with Belgium-optimised colours."""
    vv, vh = _load_on_grid(item, ref_grid)
    rgb = rtc_composite(vv, vh, co_pol_range=suggested_co,
                        cross_pol_range=suggested_cross)
    del vv, vh
    return rgb, item.label

if data:
    gif_path2 = os.path.join(GIF_DIR, "belgium_optimised.gif")
    save_timeseries_gif_lazy(
        data, gif_path2,
        composite_fn=_optimised_composite,
        title="Belgium — Optimised RTC Colour Composite",
        pixel_size_m=data[0].pixel_size_m if data[0].pixel_size_m else None,
        fps=2,
    )


# ════════════════════════════════════════════════════════════════════════════
# 9. Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Default co-pol range:     {DEFAULT_CO_POL}")
print(f"  Default cross-pol range:  {DEFAULT_CROSS_POL}")
print(f"  Suggested co-pol range:   {suggested_co}")
print(f"  Suggested cross-pol range:{suggested_cross}")
print(f"\n  To use in code:")
print(f"    rtc_composite(vv, vh,")
print(f"                  co_pol_range={suggested_co},")
print(f"                  cross_pol_range={suggested_cross})")
print(f"\nAll outputs written to: {OUT_DIR}")
