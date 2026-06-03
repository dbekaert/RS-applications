"""
NISAR GCOV Colour Composite over Belgium.

Searches for NISAR L2 GCOV products over Belgium (public Beta + Early Adopter
archive), downloads the first available HDF5 granule, extracts the diagonal
covariance terms (HHHH/HVHV for L-band; VVVV/VHVH for S-band), and produces
false-colour composites with P2–P98 colour-stretch analysis.

Usage:
    conda activate RS-applications
    python run_gcov_colorcomposite_be.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                         [--no-ea] [--limit N] [--out-dir PATH]

Output files (in OUT_DIR):
    gcov_composite.png          – false-colour RGB (R=co-pol, G=cross-pol, B=ratio)
    gcov_amplitude_hist.png     – amplitude distribution histograms
    gcov_stretch_comparison.png – three side-by-side stretch options
"""

from __future__ import annotations

import argparse
import netrc
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

from rs_tools.archives.nasa import _NISAR_EA_L2
from rs_tools.config import BoundingBox, SearchConfig
from rs_tools.datasets.catalog import get as get_dataset
from rs_tools.search import search_archive

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_START   = "2025-01-01"
DEFAULT_END     = "2026-05-28"
DEFAULT_OUT_DIR = os.path.expanduser("~/RS_applications/Applications/GCOV/BelgiumColors")
DEFAULT_LIMIT   = 20

# Belgium bounding box
BBOX_BELGIUM = BoundingBox(west=2.54, south=49.50, east=6.41, north=51.50)

# NISAR GCOV HDF5 paths — diagonal covariance (power) and geolocation
_LBAND_COPOL = "/science/LSAR/GCOV/grids/frequencyA/HHHH"
_LBAND_XPOL  = "/science/LSAR/GCOV/grids/frequencyA/HVHV"
_LBAND_X     = "/science/LSAR/GCOV/grids/frequencyA/xCoordinates"
_LBAND_Y     = "/science/LSAR/GCOV/grids/frequencyA/yCoordinates"
_LBAND_PROJ  = "/science/LSAR/GCOV/grids/frequencyA/projection"

_SBAND_COPOL = "/science/SSAR/GCOV/grids/frequencyA/VVVV"
_SBAND_XPOL  = "/science/SSAR/GCOV/grids/frequencyA/VHVH"
_SBAND_X     = "/science/SSAR/GCOV/grids/frequencyA/xCoordinates"
_SBAND_Y     = "/science/SSAR/GCOV/grids/frequencyA/yCoordinates"
_SBAND_PROJ  = "/science/SSAR/GCOV/grids/frequencyA/projection"


# ════════════════════════════════════════════════════════════════════════════
# Auth & download
# ════════════════════════════════════════════════════════════════════════════

def _get_auth() -> tuple[str | None, str | None]:
    """Return (username, password) from ~/.netrc for urs.earthdata.nasa.gov."""
    try:
        nrc = netrc.netrc()
        auth = nrc.authenticators("urs.earthdata.nasa.gov")
        if auth:
            return auth[0], auth[2]
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return None, None


def download_file(url: str, dest_dir: str, chunk_size: int = 1024 * 1024) -> str:
    """Download *url* into *dest_dir*, returning the local path.

    Skips download if the file already exists.  Follows NASA Earthdata OAuth
    redirects via ~/.netrc credentials.
    """
    fname = os.path.basename(url.split("?")[0])
    local = os.path.join(dest_dir, fname)
    if os.path.exists(local):
        print(f"  Already on disk: {local}")
        return local

    user, pw = _get_auth()
    session = requests.Session()
    if user:
        session.auth = (user, pw)

    print(f"  Downloading {fname} …", end=" ", flush=True)
    resp = session.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    size = 0
    with open(local, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            size += len(chunk)
    print(f"{size / 1e6:.1f} MB  →  {local}")
    return local


# ════════════════════════════════════════════════════════════════════════════
# HDF5 helpers
# ════════════════════════════════════════════════════════════════════════════

def print_h5_tree(path: str, max_depth: int = 4) -> None:
    """Print the HDF5 group/dataset hierarchy up to *max_depth* levels."""
    def _visit(name, obj):
        depth = name.count("/")
        if depth > max_depth:
            return
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}[DS] {name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}[G]  {name}/")

    with h5py.File(path, "r") as f:
        print(f"Root groups: {list(f.keys())}\n")
        f.visititems(_visit)


def load_gcov_channels(
    h5_path: str,
    subsample: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str | None, str]:
    """Load co-pol + cross-pol power layers from a NISAR GCOV HDF5 file.

    Prefers L-band frequencyA (HHHH, HVHV); falls back to S-band (VVVV, VHVH).

    Parameters
    ----------
    h5_path : str
        Path to the NISAR GCOV HDF5 file.
    subsample : int
        Read every *subsample*-th row and column.  Default 4 reduces a
        ~5000×5000 frame (20 m native) to ~1250×1250 (80 m effective),
        cutting memory use by ~16× while retaining full visual quality
        for colour composites.  Set to 1 for full resolution.

    Returns
    -------
    co_power, xp_power : ndarray float32
        Backscatter power arrays (negative = fill, set to NaN).
    x_coords, y_coords : ndarray or None
        Pixel-centre coordinates (map units), also subsampled.
    proj_wkt : str or None
        WKT projection string.
    band_label : str
        Human-readable label, e.g. ``"L-band (HH/HV)"``.
    """
    s = slice(None, None, subsample)  # e.g. ::4
    with h5py.File(h5_path, "r") as f:
        if _LBAND_COPOL in f and _LBAND_XPOL in f:
            co   = f[_LBAND_COPOL][s, s].astype(np.float32)
            xp   = f[_LBAND_XPOL][s, s].astype(np.float32)
            xc   = f[_LBAND_X][s] if _LBAND_X in f else None
            yc   = f[_LBAND_Y][s] if _LBAND_Y in f else None
            prj  = f[_LBAND_PROJ][()] if _LBAND_PROJ in f else None
            band_label = "L-band (HH/HV)"
        elif _SBAND_COPOL in f and _SBAND_XPOL in f:
            co   = f[_SBAND_COPOL][s, s].astype(np.float32)
            xp   = f[_SBAND_XPOL][s, s].astype(np.float32)
            xc   = f[_SBAND_X][s] if _SBAND_X in f else None
            yc   = f[_SBAND_Y][s] if _SBAND_Y in f else None
            prj  = f[_SBAND_PROJ][()] if _SBAND_PROJ in f else None
            band_label = "S-band (VV/VH)"
        else:
            raise ValueError(
                "Neither L-band HHHH/HVHV nor S-band VVVV/VHVH found in "
                f"{h5_path}.  Run print_h5_tree() to inspect the structure."
            )

    # Mask invalid/fill values (negative power is unphysical)
    co[co < 0] = np.nan
    xp[xp < 0] = np.nan

    if isinstance(prj, bytes):
        prj = prj.decode()

    eff_res = f"~{20 * subsample} m effective" if subsample > 1 else "native 20 m"
    print(f"  Loaded {band_label}  (subsample={subsample}, {eff_res})")
    print(f"    co-pol    shape={co.shape}  valid={np.sum(np.isfinite(co)):,}")
    print(f"    cross-pol shape={xp.shape}  valid={np.sum(np.isfinite(xp)):,}")
    return co, xp, xc, yc, prj, band_label


# ════════════════════════════════════════════════════════════════════════════
# Composite defaults  — reverse-engineered from the official NISAR browse image
# (granule NISAR_L2_PR_GCOV_010_066_D_064_2005_DHDH_A_20260113T…, Jan 2026)
# Convention: R = B = co-pol amplitude,  G = cross-pol amplitude
# (identical to the OPERA/ASF S1 RTC browse convention)
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_CO_RANGE = (0.047, 0.590)   # HH amplitude — reverse-engineered from NISAR browse
_DEFAULT_XP_RANGE = (0.021, 0.297)   # HV amplitude — reverse-engineered from NISAR browse


# ════════════════════════════════════════════════════════════════════════════
# Composite
# ════════════════════════════════════════════════════════════════════════════

def gcov_composite(
    co_power: np.ndarray,
    xp_power: np.ndarray,
    co_range: tuple[float, float] = _DEFAULT_CO_RANGE,
    xp_range: tuple[float, float] = _DEFAULT_XP_RANGE,
) -> np.ndarray:
    """Build a false-colour RGB composite from GCOV covariance power layers.

    Follows the official NISAR / OPERA convention (verified by reverse-engineering
    the ASF NISAR browse product):

    * **R** = co-pol amplitude (√HHHH or √VVVV), stretched to [0, 1]
    * **G** = cross-pol amplitude (√HVHV or √VHVH), stretched to [0, 1]
    * **B** = co-pol amplitude (same as R)

    This is identical to the OPERA RTC-S1 convention in
    ``rs_tools/visualization/rtc_composite.py``.
    NaN pixels (fill / out-of-swath) are rendered white.

    Parameters
    ----------
    co_power, xp_power : ndarray float32
        Covariance diagonal power layers (fill / negative → NaN).
    co_range : (float, float)
        Amplitude range used to stretch the co-pol channel (R and B).
    xp_range : (float, float)
        Amplitude range used to stretch the cross-pol channel (G).

    Returns
    -------
    rgb : ndarray (H, W, 3) float32 in [0, 1]
    """
    def _stretch(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Linear stretch to [0,1]; NaN propagates and is filled white at end."""
        return np.clip(
            (arr.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo),
            0.0, 1.0,
        )

    co_amp = np.sqrt(np.where(np.isnan(co_power), np.nan, np.clip(co_power, 0, None)))
    xp_amp = np.sqrt(np.where(np.isnan(xp_power), np.nan, np.clip(xp_power, 0, None)))

    R = _stretch(co_amp, co_range[0], co_range[1])
    G = _stretch(xp_amp, xp_range[0], xp_range[1])
    B = R.copy()   # B = co-pol (same as R) — OPERA/NISAR convention

    rgb = np.stack([R, G, B], axis=-1)
    rgb[np.isnan(rgb)] = 1.0  # NaN → white (burst edges / out-of-swath)
    return rgb


def compute_saturation(
    co_power: np.ndarray,
    xp_power: np.ndarray,
    co_range: tuple[float, float],
    xp_range: tuple[float, float],
    label: str = "",
) -> tuple[float, float]:
    """Report fraction of valid amplitude pixels saturated (clipped at 0 or 1).

    Mirrors the saturation analysis in run_colorcomposite_be.py for S1.
    """
    co_amp = np.sqrt(co_power[np.isfinite(co_power) & (co_power > 0)])
    xp_amp = np.sqrt(xp_power[np.isfinite(xp_power) & (xp_power > 0)])
    total  = len(co_amp) + len(xp_amp)

    sat_lo = int(np.sum(co_amp < co_range[0])) + int(np.sum(xp_amp < xp_range[0]))
    sat_hi = int(np.sum(co_amp > co_range[1])) + int(np.sum(xp_amp > xp_range[1]))

    pct_lo  = 100.0 * sat_lo  / total if total else 0.0
    pct_hi  = 100.0 * sat_hi  / total if total else 0.0
    pct_tot = pct_lo + pct_hi
    print(f"\nSaturation{' — ' + label if label else ''}:")
    print(f"  Co-pol  range : {co_range}")
    print(f"  Cross-pol range: {xp_range}")
    print(f"  Saturated low : {pct_lo:.2f}%")
    print(f"  Saturated high: {pct_hi:.2f}%")
    print(f"  Total saturated: {pct_tot:.2f}%")
    return pct_lo, pct_hi


# ════════════════════════════════════════════════════════════════════════════
# Amplitude statistics & colour-range suggestion
# ════════════════════════════════════════════════════════════════════════════

def analyse_amplitude_stats(
    co_power: np.ndarray,
    xp_power: np.ndarray,
    co_lbl: str = "co-pol",
    xp_lbl: str = "cross-pol",
    tag: str = "",
    max_pixels: int = 5_000_000,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute amplitude (√power) percentile statistics and suggest colour ranges.

    Returns
    -------
    co_range, xp_range : (float, float)
        P2–P98 amplitude ranges for co-pol and cross-pol.
    """
    rng = np.random.default_rng(42)

    co_amp = np.sqrt(co_power[np.isfinite(co_power) & (co_power > 0)])
    xp_amp = np.sqrt(xp_power[np.isfinite(xp_power) & (xp_power > 0)])

    if len(co_amp) > max_pixels:
        co_amp = co_amp[rng.choice(len(co_amp), max_pixels, replace=False)]
    if len(xp_amp) > max_pixels:
        xp_amp = xp_amp[rng.choice(len(xp_amp), max_pixels, replace=False)]

    percentiles = [1, 2, 5, 25, 50, 75, 95, 98, 99]
    print(f"\n{'='*60}")
    print(f"Amplitude statistics — NISAR GCOV{' | ' + tag if tag else ''}")
    print(f"{'='*60}")
    print(f"Valid pixels:  co-pol={len(co_amp):,}  cross-pol={len(xp_amp):,}")
    print(f"\n{'Pct':>5}  {co_lbl:>15}  {xp_lbl:>15}")
    for p in percentiles:
        print(f"  P{p:<3d}  {np.percentile(co_amp, p):>15.5f}  {np.percentile(xp_amp, p):>15.5f}")
    print(f"  Mean  {np.mean(co_amp):>15.5f}  {np.mean(xp_amp):>15.5f}")
    print(f"  Std   {np.std(co_amp):>15.5f}  {np.std(xp_amp):>15.5f}")

    co_range = (float(np.percentile(co_amp, 2)), float(np.percentile(co_amp, 98)))
    xp_range = (float(np.percentile(xp_amp, 2)), float(np.percentile(xp_amp, 98)))
    print(f"\nSuggested colour ranges (P2–P98):")
    print(f"  {co_lbl:15s}: {co_range}")
    print(f"  {xp_lbl:15s}: {xp_range}")
    return co_range, xp_range, co_amp, xp_amp


# ════════════════════════════════════════════════════════════════════════════
# Plot helpers
# ════════════════════════════════════════════════════════════════════════════

def plot_amplitude_histograms(
    co_amp: np.ndarray,
    xp_amp: np.ndarray,
    co_range: tuple[float, float],
    xp_range: tuple[float, float],
    co_lbl: str,
    xp_lbl: str,
    granule_id: str,
    out_path: str,
    default_co_range: tuple[float, float] | None = None,
    default_xp_range: tuple[float, float] | None = None,
) -> None:
    """Save amplitude histograms with default and P2-P98 ranges overlaid.

    When *default_co_range* / *default_xp_range* are supplied the plot shows
    both the current hard-coded defaults (red) and the data-driven P2-P98
    suggestion (lime), mirroring the S1 saturation-analysis histogram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, amp, rng, def_rng, pol, color in [
        (axes[0], co_amp,  co_range, default_co_range, co_lbl, "steelblue"),
        (axes[1], xp_amp, xp_range, default_xp_range, xp_lbl, "darkorange"),
    ]:
        ax.hist(amp, bins=300, density=True, alpha=0.7, color=color,
                label=f"{pol} amplitude")
        if def_rng is not None:
            ax.axvline(def_rng[0], color="red", ls="--", lw=1.5,
                       label=f"Default min={def_rng[0]:.4f}")
            ax.axvline(def_rng[1], color="red", ls="-",  lw=1.5,
                       label=f"Default max={def_rng[1]:.4f}")
        ax.axvline(rng[0], color="lime", ls="--", lw=1.5, label=f"P2  = {rng[0]:.4f}")
        ax.axvline(rng[1], color="lime", ls="-",  lw=1.5, label=f"P98 = {rng[1]:.4f}")
        ax.set_title(f"{pol} amplitude", fontsize=11)
        ax.set_xlabel("Amplitude (√power)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.suptitle(f"NISAR GCOV amplitude distributions  —  {granule_id}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_composite(
    co_power: np.ndarray,
    xp_power: np.ndarray,
    co_range: tuple[float, float],
    xp_range: tuple[float, float],
    band_label: str,
    co_lbl: str,
    xp_lbl: str,
    granule_id: str,
    out_path: str,
) -> None:
    """Save a single false-colour composite PNG."""
    rgb = gcov_composite(co_power, xp_power, co_range=co_range, xp_range=xp_range)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    ax.set_title(
        f"NISAR GCOV false-colour composite — {band_label}\n"
        f"{granule_id}\n"
        f"R=B={co_lbl} amp  G={xp_lbl} amp  (OPERA/NISAR convention)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_stretch_comparison(
    co_power: np.ndarray,
    xp_power: np.ndarray,
    co_amp: np.ndarray,
    xp_amp: np.ndarray,
    band_label: str,
    granule_id: str,
    out_path: str,
) -> None:
    """Save a three-panel stretch comparison PNG."""
    stretch_options = [
        ("P2–P98 (auto)",
         (float(np.percentile(co_amp, 2)),  float(np.percentile(co_amp, 98))),
         (float(np.percentile(xp_amp, 2)),  float(np.percentile(xp_amp, 98)))),
        ("Tight (P5–P95)",
         (float(np.percentile(co_amp, 5)),  float(np.percentile(co_amp, 95))),
         (float(np.percentile(xp_amp, 5)),  float(np.percentile(xp_amp, 95)))),
        ("Wide  (P1–P99)",
         (float(np.percentile(co_amp, 1)),  float(np.percentile(co_amp, 99))),
         (float(np.percentile(xp_amp, 1)),  float(np.percentile(xp_amp, 99)))),
    ]
    pol_labels = band_label.split("(")[1].rstrip(")").split("/")
    co_lbl = pol_labels[0].strip()
    xp_lbl = pol_labels[1].strip()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (label, cr, xr) in zip(axes, stretch_options):
        rgb_i = gcov_composite(co_power, xp_power, co_range=cr, xp_range=xr)
        ax.imshow(rgb_i, origin="upper")
        ax.set_axis_off()
        ax.set_title(
            f"{label}\n"
            f"{co_lbl}: [{cr[0]:.3f}, {cr[1]:.3f}]\n"
            f"{xp_lbl}: [{xr[0]:.3f}, {xr[1]:.3f}]",
            fontsize=9,
        )
    plt.suptitle(
        f"NISAR GCOV — colour stretch comparison  |  {band_label}", fontsize=12
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # ── 1. Dataset info ──────────────────────────────────────────────────────
    gcov_ds = get_dataset("NISAR_L2_GCOV")
    print(f"\nDataset : {gcov_ds.name}")
    print(f"  resolution : {gcov_ds.spatial_resolution}")
    print(f"  EA L2 coll : {_NISAR_EA_L2}")

    # ── 2. Search ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Searching NISAR GCOV over Belgium …")
    print(f"  period     : {args.start} → {args.end}")
    print(f"  include_ea : {not args.no_ea}")
    config = SearchConfig(
        start_date=args.start,
        end_date=args.end,
        bbox=BBOX_BELGIUM,
        collections=["NISAR_L2_GCOV"],
        limit=args.limit,
        include_ea=not args.no_ea,
    )
    items = search_archive("nasa", config)
    print(f"\nFound {len(items)} GCOV granule(s).")

    if not items:
        print(
            "\nNo GCOV products found over Belgium in the requested period.\n"
            "Tips:\n"
            "  • Expand the date range (--start / --end)\n"
            "  • Ensure EA credentials are in ~/.netrc (machine urs.earthdata.nasa.gov)\n"
            "  • Run without --no-ea to include the Early Adopter collection\n"
        )
        sys.exit(0)

    for i, it in enumerate(items[:5]):
        props = it.get("properties", {})
        print(
            f"  [{i}] {it['id']}"
            f"  datetime={props.get('datetime', 'n/a')}"
            f"  platform={props.get('platform', 'n/a')}"
        )

    # ── 3. Pick first granule ────────────────────────────────────────────────
    item = items[0]
    granule_id = item["id"]
    print(f"\nSelected granule: {granule_id}")

    assets = item.get("assets", {})
    h5_asset = assets.get("data") or next(iter(assets.values()), None)
    if h5_asset is None:
        print("ERROR: no download URL found in item assets.")
        sys.exit(1)
    h5_url = h5_asset["href"]
    print(f"  URL: {h5_url}")

    # ── 4. Download ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Downloading HDF5 …")
    h5_path = download_file(h5_url, out_dir)

    # ── 5. Inspect structure ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HDF5 structure (depth ≤ 4):")
    print_h5_tree(h5_path, max_depth=4)

    # ── 6. Load covariance layers ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading covariance layers …")
    co_power, xp_power, x_coords, y_coords, proj_wkt, band_label = (
        load_gcov_channels(h5_path, subsample=args.subsample)
    )

    # Polarisation labels derived from band_label, e.g. "L-band (HH/HV)"
    pol_parts = band_label.split("(")[1].rstrip(")").split("/")
    co_lbl = pol_parts[0].strip()
    xp_lbl = pol_parts[1].strip()

    # ── 7. Amplitude statistics & colour ranges ──────────────────────────────
    co_range, xp_range, co_amp, xp_amp = analyse_amplitude_stats(
        co_power, xp_power,
        co_lbl=co_lbl, xp_lbl=xp_lbl,
        tag=granule_id,
    )

    # ── 7b. Saturation comparison (default hard-coded vs P2–P98 suggestion) ──
    print(f"\n{'='*60}")
    print("Saturation analysis …")
    compute_saturation(co_power, xp_power, _DEFAULT_CO_RANGE, _DEFAULT_XP_RANGE,
                       label="current defaults")
    compute_saturation(co_power, xp_power, co_range, xp_range,
                       label="P2–P98 (Belgium suggested)")

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving figures …")

    plot_amplitude_histograms(
        co_amp, xp_amp,
        co_range, xp_range,
        co_lbl, xp_lbl,
        granule_id,
        os.path.join(out_dir, "gcov_amplitude_hist.png"),
        default_co_range=_DEFAULT_CO_RANGE,
        default_xp_range=_DEFAULT_XP_RANGE,
    )

    plot_composite(
        co_power, xp_power,
        co_range, xp_range,
        band_label, co_lbl, xp_lbl,
        granule_id,
        os.path.join(out_dir, "gcov_composite.png"),
    )

    plot_stretch_comparison(
        co_power, xp_power,
        co_amp, xp_amp,
        band_label,
        granule_id,
        os.path.join(out_dir, "gcov_stretch_comparison.png"),
    )

    # ── 9. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Granule     : {granule_id}")
    print(f"  Band        : {band_label}")
    print(f"  Co-pol ({co_lbl}) range   : {co_range}")
    print(f"  Cross-pol ({xp_lbl}) range : {xp_range}")
    print(f"\n  gcov_composite(co_power, x_power,")
    print(f"                 co_range={co_range},")
    print(f"                 xp_range={xp_range})")
    print(f"\n  Output: {out_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NISAR GCOV false-colour composite over Belgium.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start",   default=DEFAULT_START,   help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",     default=DEFAULT_END,     help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-ea",   action="store_true",     help="Exclude Early Adopter collection")
    parser.add_argument("--limit",   type=int, default=DEFAULT_LIMIT, help="Max search results")
    parser.add_argument("--out-dir",   default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--subsample", type=int, default=4,
                        help="Read every N-th row/col (default 4 → ~80 m for 20 m native)")
    args = parser.parse_args()
    main(args)
