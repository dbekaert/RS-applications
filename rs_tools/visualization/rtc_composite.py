"""OPERA RTC-S1 browse image compositing.

Creates false-colour RGB composites from co-pol (VV) and cross-pol (VH)
SAR backscatter.

Default colour convention (Belgium-optimised):
    R = sqrt(VV),   range [0.129, 0.871]
    G = sqrt(VH),   range [0.040, 0.358]
    B = sqrt(VV)    (same as R)

Derived from a statistical analysis of ~66 monthly OPERA RTC-S1 passes
over Belgium, using P2/P98 percentiles.  Reduces saturation from ~15 %
(ASF/HyP3 defaults) to ~4 % while preserving contrast.

Named presets are available via :data:`PRESETS`.

Reference (original ASF convention):
    https://github.com/ASFHyP3/opera-rtc-s1-browse
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── Named colour presets ────────────────────────────────────────────────────
# Each preset maps to (co_pol_range, cross_pol_range) in amplitude space.
PRESETS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    # Belgium-optimised (P2/P98), derived from 66 monthly passes 2019–2025.
    # Wider dynamic range → less saturation, better urban/farmland contrast.
    "default": ((0.129, 0.871), (0.040, 0.358)),
    # Original ASF / HyP3 convention — good for global overviews.
    "OPERA_global": ((0.14, 0.52), (0.05, 0.259)),
}

# Active default ranges (used when callers pass no explicit ranges).
_CO_POL_RANGE, _CROSS_POL_RANGE = PRESETS["default"]


def _normalize_band(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Scale array to [0, 1]; NaN pixels remain NaN (filled to white later)."""
    out = (arr.astype(np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def rtc_composite(
    vv: np.ndarray,
    vh: np.ndarray,
    co_pol_range: Tuple[float, float] = _CO_POL_RANGE,
    cross_pol_range: Tuple[float, float] = _CROSS_POL_RANGE,
    preset: Optional[str] = None,
) -> np.ndarray:
    """Create a false-colour RGB composite from RTC VV/VH power arrays.

    The input arrays are expected to be in **linear power** scale
    (not dB).  They are converted to amplitude (sqrt) and normalised.

    Parameters
    ----------
    vv : np.ndarray
        2-D co-pol (VV) backscatter array in linear power.
    vh : np.ndarray
        2-D cross-pol (VH) backscatter array in linear power.
    co_pol_range : tuple[float, float]
        (min, max) amplitude range for VV normalisation.
    cross_pol_range : tuple[float, float]
        (min, max) amplitude range for VH normalisation.
    preset : str, optional
        Named colour preset from :data:`PRESETS`.  When given,
        overrides *co_pol_range* and *cross_pol_range*.
        E.g. ``"OPERA_global"`` for the original ASF/HyP3 ranges.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` float RGB array with values in [0, 1].
    """
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. "
                f"Available: {sorted(PRESETS)}"
            )
        co_pol_range, cross_pol_range = PRESETS[preset]
    vv_amp = np.sqrt(np.where(np.isnan(vv), np.nan, np.clip(vv, 0, None)))
    vh_amp = np.sqrt(np.where(np.isnan(vh), np.nan, np.clip(vh, 0, None)))

    r = _normalize_band(vv_amp, *co_pol_range)
    g = _normalize_band(vh_amp, *cross_pol_range)
    b = r.copy()  # B = VV same as R

    rgb = np.dstack([r, g, b])
    # Fill NoData (NaN) with white rather than black so that burst
    # boundaries and AOI edges outside the data footprint are white.
    rgb[np.isnan(rgb)] = 1.0
    return rgb


def plot_rtc_composite(
    vv: np.ndarray,
    vh: np.ndarray,
    title: Optional[str] = None,
    co_pol_range: Tuple[float, float] = _CO_POL_RANGE,
    cross_pol_range: Tuple[float, float] = _CROSS_POL_RANGE,
    preset: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Create and display an RTC false-colour composite.

    Parameters
    ----------
    vv, vh : np.ndarray
        2-D backscatter arrays in linear power.
    title : str | None
        Plot title.
    co_pol_range, cross_pol_range : tuple
        Amplitude scaling ranges.
    preset : str, optional
        Named colour preset (overrides ranges).
    figsize : tuple
        Figure size (only used when *ax* is None).
    ax : matplotlib.axes.Axes | None
        Existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rgb = rtc_composite(vv, vh, co_pol_range, cross_pol_range, preset=preset)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig
