"""
Geocode SLC bursts using embedded GCPs, crop to Deurganckdock AOI,
with multi-looking for speckle reduction, and produce before/after
comparison + animated GIF.

Multi-look strategy for oil slick detection:
  - IW SLC pixel spacing: ~2.3 m (range) x ~14 m (azimuth)
  - Resolution: ~5 m (range) x ~20 m (azimuth)
  - Heavy multi-look 3 (az) x 9 (rg) on INTENSITY (not dB!) →
    ~21 m x 42 m → 27 equivalent looks
  - Plus post-geocoding boxcar (5x5) for further smoothing
  - Effective resolution ~100-150 m — still fine for 1.5 km dock
  - The Deurganckdock is heavily contaminated by sidelobes from port
    infrastructure (cranes, ships, containers). Aggressive filtering
    is needed to suppress these and reveal the oil dampening signal.
"""
import os
import numpy as np
import rasterio
import rasterio.control
from osgeo import gdal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
BURST_DIR = os.path.join(OUT_DIR, "bursts")

before_tiff = os.path.join(
    BURST_DIR,
    "before_same_orbit_2026-04-04/S1_343971_IW2_20260404T173317_VV_A801-BURST.tiff",
)
after_tiff = os.path.join(
    BURST_DIR,
    "after_2026-04-10/S1_343971_IW2_20260410T173219_VV_134F-BURST.tiff",
)

# AOI: Wider view including Scheldt river and dock surroundings
# to see oil spreading beyond the dock into open water where
# sidelobe contamination is lower
BUFFER = 0.020  # ~2 km buffer
AOI = {
    "west": 4.251283836960951 - BUFFER,
    "south": 51.28632393164611 - BUFFER,
    "east": 4.271149638243543 + BUFFER,
    "north": 51.298511604339325 + BUFFER,
}

# Target resolution ~12 m after multi-look (at lat 51°: 0.00011° ≈ 12 m)
RES_DEG = 0.00011

# Moderate multi-look: 1 (az) x 5 (rg) = 5 looks
# Resolution: ~14 m (az) x ~12 m (rg) → speckle reduced by sqrt(5) ≈ 2.2x
ML_AZ = 1   # azimuth (rows) — already ~14 m spacing
ML_RG = 5   # range (cols) — 2.3 m * 5 ≈ 11.5 m

# Light post-geocoding boxcar smoothing
BOXCAR = 3  # 3x3 → ~36 m at 12 m pixel spacing

print("=" * 60)
print("Geocoding SLC bursts using embedded GCPs")
print("=" * 60)


def multilook_intensity(data, ml_az, ml_rg):
    """
    Multi-look complex SLC data by averaging intensity over a window.

    Multi-looking must be done on intensity (|z|^2), not amplitude or dB,
    because SAR speckle is multiplicative on intensity. Averaging intensity
    is the maximum-likelihood estimator for the underlying reflectivity.

    This also suppresses azimuth/range sidelobes from point scatterers
    (ships, cranes) because the sidelobe energy is spread and averaged.
    """
    if np.iscomplexobj(data):
        intensity = (np.abs(data) ** 2).astype(np.float64)
    else:
        intensity = data.astype(np.float64) ** 2

    nrows, ncols = intensity.shape
    # Trim to exact multiple of window size
    nr = (nrows // ml_az) * ml_az
    nc = (ncols // ml_rg) * ml_rg
    intensity = intensity[:nr, :nc]

    # Reshape and average
    ml = intensity.reshape(nr // ml_az, ml_az, nc // ml_rg, ml_rg).mean(axis=(1, 3))
    return ml.astype(np.float32)


def boxcar_smooth(arr, win):
    """
    NaN-aware boxcar (uniform) smoothing using cumulative sums.
    More efficient than scipy.ndimage.uniform_filter and handles NaN.
    """
    if win <= 1:
        return arr
    pad = win // 2
    # Replace NaN with 0 for summation, track valid pixel count
    mask = np.isfinite(arr)
    filled = np.where(mask, arr, 0.0)
    count = mask.astype(np.float64)

    # 2D cumsum-based boxcar
    def cumsum_box(a, w):
        p = w // 2
        padded = np.pad(a, p, mode="constant", constant_values=0)
        cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        r = cs[w:, w:] - cs[:-w, w:] - cs[w:, :-w] + cs[:-w, :-w]
        # Trim to original size
        return r[:a.shape[0], :a.shape[1]]

    s = cumsum_box(filled.astype(np.float64), win)
    c = cumsum_box(count, win)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (s / c).astype(np.float32)
    result[c < 1] = np.nan
    return result


def geocode_burst_to_aoi(src_path, aoi, res_deg, label="", ml_az=1, ml_rg=5):
    """
    Complex SLC → multi-look intensity → dB → GDAL TPS warp via GCPs → EPSG:4326.

    Multi-looking is applied in radar geometry on intensity to reduce speckle
    and suppress point-scatterer sidelobes while preserving oil slick features.
    """
    print(f"\n  [{label}] Reading: {os.path.basename(src_path)}")

    with rasterio.open(src_path) as src:
        data = src.read(1)
        gcps_orig = src.gcps[0]
        gcps_crs = src.gcps[1]
        height, width = src.height, src.width

    print(f"  [{label}] Original shape: {height} x {width}")

    # Multi-look on intensity in radar geometry
    ml_intensity = multilook_intensity(data, ml_az, ml_rg)
    ml_h, ml_w = ml_intensity.shape
    print(f"  [{label}] Multi-looked {ml_az}x{ml_rg} → {ml_h} x {ml_w}  ({ml_az * ml_rg} looks)")

    # Rescale GCPs to multi-looked pixel grid
    gcps = []
    for g in gcps_orig:
        gcps.append(rasterio.control.GroundControlPoint(
            row=g.row / ml_az,
            col=g.col / ml_rg,
            x=g.x,
            y=g.y,
            z=g.z,
        ))

    # Convert to dB
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(ml_intensity)
    db[~np.isfinite(db)] = 0.0  # GDAL needs finite values

    print(f"  [{label}] dB range: {np.min(db):.1f} to {np.max(db):.1f}")

    # Write temp GeoTIFF with multi-looked intensity dB + rescaled GCPs
    tmp_path = os.path.join(OUT_DIR, f"_tmp_amp_{label}.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": ml_w,
        "height": ml_h,
        "count": 1,
        "crs": None,
        "transform": rasterio.transform.from_bounds(0, ml_h, ml_w, 0, ml_w, ml_h),
    }
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(db, 1)
        dst.gcps = (gcps, gcps_crs)

    print(f"  [{label}] Wrote temp amplitude TIFF with {len(gcps)} GCPs")

    # GDAL Warp with TPS → regular EPSG:4326 grid, cropped to AOI
    geo_path = os.path.join(OUT_DIR, f"geocoded_{label}.tif")
    ds = gdal.Warp(
        geo_path,
        tmp_path,
        dstSRS="EPSG:4326",
        outputBounds=[aoi["west"], aoi["south"], aoi["east"], aoi["north"]],
        xRes=res_deg,
        yRes=res_deg,
        resampleAlg="bilinear",
        tps=True,
        dstNodata=np.nan,
    )
    ds.FlushCache()
    ds = None
    os.remove(tmp_path)

    # Read back geocoded result
    with rasterio.open(geo_path) as src:
        geocoded = src.read(1)
        transform = src.transform
        bounds = src.bounds
        print(f"  [{label}] Geocoded shape: {geocoded.shape}")
        print(f"  [{label}] Bounds: W={bounds.left:.4f} S={bounds.bottom:.4f} E={bounds.right:.4f} N={bounds.top:.4f}")
        print(f"  [{label}] Pixel size: {abs(transform.a) * 111000:.1f} m (lon) x {abs(transform.e) * 111000:.1f} m (lat)")

    geocoded[geocoded == 0.0] = np.nan
    return geocoded, transform, bounds


before_geo, before_tf, before_bounds = geocode_burst_to_aoi(
    before_tiff, AOI, RES_DEG, "before", ml_az=ML_AZ, ml_rg=ML_RG,
)
after_geo, after_tf, after_bounds = geocode_burst_to_aoi(
    after_tiff, AOI, RES_DEG, "after", ml_az=ML_AZ, ml_rg=ML_RG,
)

# Ensure same dimensions
min_r = min(before_geo.shape[0], after_geo.shape[0])
min_c = min(before_geo.shape[1], after_geo.shape[1])
before_geo = before_geo[:min_r, :min_c]
after_geo = after_geo[:min_r, :min_c]

print(f"\n  Final cropped shape: {before_geo.shape}")

# Apply post-geocoding boxcar smoothing
print(f"  Applying {BOXCAR}x{BOXCAR} boxcar smoothing (~{BOXCAR * 30}m) ...")
before_smooth = boxcar_smooth(before_geo, BOXCAR)
after_smooth = boxcar_smooth(after_geo, BOXCAR)

# Common display range
all_valid = np.concatenate([
    before_smooth[np.isfinite(before_smooth)],
    after_smooth[np.isfinite(after_smooth)],
])
vmin = float(np.nanpercentile(all_valid, 2))
vmax = float(np.nanpercentile(all_valid, 98))
print(f"  Display range: {vmin:.1f} to {vmax:.1f} dB")

extent = [AOI["west"], AOI["east"], AOI["south"], AOI["north"]]

# Dock outline for reference (approximate corners of Deurganckdock water area)
DOCK_LON = [4.2513, 4.2711, 4.2711, 4.2513, 4.2513]
DOCK_LAT = [51.2863, 51.2863, 51.2985, 51.2985, 51.2863]

eff_res_m = int(BOXCAR * RES_DEG * 111000)

# ── Side-by-side geocoded comparison ──────────────────────────
print("\n  Creating side-by-side comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

for ax, data, lbl in [
    (axes[0], before_smooth, "Before — 2026-04-04"),
    (axes[1], after_smooth, "After — 2026-04-10"),
]:
    im = ax.imshow(
        data, cmap="gray", vmin=vmin, vmax=vmax,
        origin="upper", extent=extent, aspect="equal",
    )
    ax.plot(DOCK_LON, DOCK_LAT, "r-", linewidth=1.5, label="Deurganckdock")
    ax.set_title(lbl, fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", fontsize=9)

fig.suptitle(
    "Deurganckdock Oil Slick \u2014 Geocoded Sentinel-1 VV (dB)\n"
    f"Same orbit 343971_IW2  |  {ML_AZ}\u00d7{ML_RG} multi-look + {BOXCAR}\u00d7{BOXCAR} boxcar  |  ~{eff_res_m} m effective",
    fontsize=13,
)
plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="VV Backscatter (dB)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "oil_slick_geocoded_sidebyside.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: oil_slick_geocoded_sidebyside.png")

# ── Difference map ────────────────────────────────────────────
print("  Creating difference map ...")
diff = after_smooth - before_smooth

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(
    diff, cmap="RdBu_r", vmin=-6, vmax=6,
    origin="upper", extent=extent, aspect="equal",
)
ax.plot(DOCK_LON, DOCK_LAT, "k-", linewidth=1.5, label="Deurganckdock")
ax.set_title(
    "Backscatter difference (dB): 2026-04-10 minus 2026-04-04\n"
    "Blue = darker after (oil dampens Bragg scattering)  |  Red = brighter",
    fontsize=12,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="upper left", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="\u0394dB")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "oil_slick_geocoded_difference.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: oil_slick_geocoded_difference.png")

# ── Animated GIF: before ↔ after ─────────────────────────────
print("  Creating animated GIF ...")
gif_frames = []
for data, lbl in [
    (before_smooth, "Before \u2014 2026-04-04"),
    (after_smooth, "After \u2014 2026-04-10  (oil slick event)"),
]:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(
        data, cmap="gray", vmin=vmin, vmax=vmax,
        origin="upper", extent=extent, aspect="equal",
    )
    ax.plot(DOCK_LON, DOCK_LAT, "r-", linewidth=1.5, label="Deurganckdock")
    ax.set_title(
        f"Deurganckdock Oil Slick \u2014 Sentinel-1 VV (dB)\n"
        f"{lbl}  |  {ML_AZ}\u00d7{ML_RG} multi-look + {BOXCAR}\u00d7{BOXCAR} boxcar",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    buf_path = os.path.join(OUT_DIR, "_tmp_frame.png")
    fig.savefig(buf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gif_frames.append(Image.open(buf_path).copy())

os.remove(buf_path)

gif_path = os.path.join(OUT_DIR, "gifs", "oil_slick_geocoded.gif")
gif_frames[0].save(
    gif_path, save_all=True, append_images=gif_frames[1:],
    duration=1500, loop=0, optimize=True,
)
print(f"  Saved: gifs/oil_slick_geocoded.gif ({os.path.getsize(gif_path) / 1024:.0f} KB)")

# ── AOI Statistics ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Geocoded AOI Statistics (smoothed)")
print("=" * 60)
vb = before_smooth[np.isfinite(before_smooth)]
va = after_smooth[np.isfinite(after_smooth)]
vd = diff[np.isfinite(diff)]
print(f"  Mean before:  {np.mean(vb):.2f} dB")
print(f"  Mean after:   {np.mean(va):.2f} dB")
print(f"  Delta:        {np.mean(va) - np.mean(vb):+.2f} dB")
print(f"  Pixels darkened >3 dB: {np.sum(vd < -3) / max(len(vd), 1) * 100:.1f}%")
print(f"  Pixels darkened >6 dB: {np.sum(vd < -6) / max(len(vd), 1) * 100:.1f}%")

print("\nDone! All geocoded outputs in:", OUT_DIR)
