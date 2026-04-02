"""Frame conversion utilities for animated visualizations.

Converts ``xarray.DataArray`` time-series to sequences of RGB frames
suitable for GIF export, including single-panel, dual-panel, and
multi-product layouts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds


def _downsample_2d(arr: np.ndarray, max_pixels: int) -> np.ndarray:
    """Downsample a 2-D float32 array so max(H, W) <= *max_pixels*.

    Uses GDAL (via rasterio) with ``Resampling.average`` so that NaN /
    nodata pixels are excluded from the averaging kernel automatically,
    preserving spatial coverage without the NaN-bleeding artefacts that
    PIL's LANCZOS filter produces.
    """
    h, w = arr.shape[:2]
    if max(h, w) <= max_pixels:
        return arr
    scale = max_pixels / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    transform = from_bounds(0, 0, w, h, w, h)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype="float32",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(arr.astype(np.float32), 1)

        with memfile.open() as src:
            result = src.read(
                1,
                out_shape=(new_h, new_w),
                resampling=Resampling.average,
            )
    return result.astype(np.float32)


def _get_cmap(name):
    """Get a colormap by name or return an existing ``Colormap`` object."""
    if isinstance(name, mpl.colors.Colormap):
        return name
    return mpl.colormaps[name]


def data_to_rgb_frames(
    data: xr.DataArray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    time_dim: str = "time",
    step: int = 1,
) -> Tuple[List[np.ndarray], List[str]]:
    """Convert a 3-D DataArray into a list of (H, W, 3) RGB frames.

    Parameters
    ----------
    data : xr.DataArray
        Array with shape ``(time, y, x)``.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float | None
        Colour-scale limits.  Computed from data if *None*.
    time_dim : str
        Name of the time dimension.
    step : int
        Sample every *step*-th time step (1 = all frames).

    Returns
    -------
    frames : list[np.ndarray]
        RGB float arrays with shape ``(H, W, 3)``, values in [0, 1].
    labels : list[str]
        ISO date string per frame.
    """
    colormap = _get_cmap(cmap)
    if vmin is None:
        vmin = float(np.nanpercentile(data.values, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data.values, 98))

    times = data[time_dim].values
    frames: List[np.ndarray] = []
    labels: List[str] = []

    for i in range(0, len(times), step):
        arr = data.isel({time_dim: i}).values
        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
        rgb = colormap(norm)[:, :, :3].astype(np.float32)
        frames.append(rgb)
        labels.append(str(np.datetime_as_string(times[i], unit="D")))

    return frames, labels


def make_colormap_composite(
    item: "LoadedItem",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    asset: Optional[str] = None,
    max_pixels: int = 2400,
) -> Tuple[np.ndarray, str]:
    """Convert a single :class:`~rs_tools.datasets.loader.LoadedItem` to an RGB frame.

    Loads data from disk, applies the colourmap, then unloads to keep
    memory usage minimal.  Intended for use as *composite_fn* with
    :func:`~rs_tools.visualization.animation.save_timeseries_gif_lazy`.

    Parameters
    ----------
    item : LoadedItem
        A dataset item (may be on disk or already loaded).
    cmap : str
        Matplotlib colormap name or ``Colormap`` object.
    vmin, vmax : float
        Colour-scale limits.
    asset : str, optional
        Asset key to use.  When *None*, the first available key is used.
    max_pixels : int
        If the raster's longest side exceeds this, downsample before
        applying the colourmap.  Set to 0 to disable.

    Returns
    -------
    rgb : np.ndarray
        ``(H, W, 3)`` float32 array with values in [0, 1].
    label : str
        Human-readable label from the item.
    """
    item.load()
    key = asset if asset is not None else next(iter(item.data))
    arr = item.data[key].values
    item.unload()

    if max_pixels and max(arr.shape[:2]) > max_pixels:
        arr = _downsample_2d(arr, max_pixels)

    colormap = _get_cmap(cmap)
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
    del arr
    rgb = colormap(norm)[:, :, :3].astype(np.float32)
    del norm
    return rgb, item.label


def make_dual_panel_composite(
    left_item: "LoadedItem",
    right_item: "LoadedItem",
    left_cmap: str = "viridis",
    right_cmap: str = "viridis",
    left_vmin: float = 0.0,
    left_vmax: float = 1.0,
    right_vmin: float = 0.0,
    right_vmax: float = 1.0,
    left_label: str = "",
    right_label: str = "",
    left_asset: Optional[str] = None,
    right_asset: Optional[str] = None,
    gap_px: int = 4,
    max_pixels: int = 2400,
) -> Tuple[np.ndarray, str]:
    """Build a side-by-side RGB frame from two :class:`LoadedItem` objects.

    Loads each item from disk, applies colormaps, concatenates
    horizontally, then unloads both.  Intended for use as
    *composite_fn* with
    :func:`~rs_tools.visualization.animation.save_timeseries_gif_lazy`.

    Parameters
    ----------
    left_item, right_item : LoadedItem
        Data items for the left and right panels.
    left_cmap, right_cmap : str
        Colormap names.
    left_vmin, left_vmax, right_vmin, right_vmax : float
        Colour-scale limits for each panel.
    left_label, right_label : str
        Product labels embedded in the returned label string.
    left_asset, right_asset : str, optional
        Asset keys.  When *None*, the first available key is used.
    gap_px : int
        White gap width between panels in pixels.

    Returns
    -------
    rgb : np.ndarray
        ``(H, W_left + gap + W_right, 3)`` float32 array.
    label : str
        Combined date + product label string.
    """
    left_item.load()
    lk = left_asset if left_asset is not None else next(iter(left_item.data))
    larr = left_item.data[lk].values
    left_item.unload()

    if max_pixels and max(larr.shape[:2]) > max_pixels:
        larr = _downsample_2d(larr, max_pixels)

    lcmap = _get_cmap(left_cmap)
    lnorm = np.clip((larr - left_vmin) / (left_vmax - left_vmin + 1e-10), 0, 1)
    del larr
    lrgb = lcmap(lnorm)[:, :, :3].astype(np.float32)
    del lnorm

    right_item.load()
    rk = right_asset if right_asset is not None else next(iter(right_item.data))
    rarr = right_item.data[rk].values
    right_item.unload()

    if max_pixels and max(rarr.shape[:2]) > max_pixels:
        rarr = _downsample_2d(rarr, max_pixels)

    rcmap = _get_cmap(right_cmap)
    rnorm = np.clip((rarr - right_vmin) / (right_vmax - right_vmin + 1e-10), 0, 1)
    del rarr
    rrgb = rcmap(rnorm)[:, :, :3].astype(np.float32)
    del rnorm

    h = max(lrgb.shape[0], rrgb.shape[0])
    if lrgb.shape[0] < h:
        pad = np.ones((h - lrgb.shape[0], lrgb.shape[1], 3), dtype=np.float32)
        lrgb = np.vstack([lrgb, pad])
    if rrgb.shape[0] < h:
        pad = np.ones((h - rrgb.shape[0], rrgb.shape[1], 3), dtype=np.float32)
        rrgb = np.vstack([rrgb, pad])

    gap = np.ones((h, gap_px, 3), dtype=np.float32)
    combined = np.hstack([lrgb, gap, rrgb])

    tag = f"{left_label} | {right_label}" if left_label or right_label else ""
    label = f"{left_item.label}  {tag}".strip()
    return combined, label


def make_overlay_composite(
    base_item: "LoadedItem",
    overlay_item: "LoadedItem",
    base_cmap: str = "YlGn",
    overlay_cmap: str = "Reds",
    base_vmin: float = 0.0,
    base_vmax: float = 1.0,
    overlay_threshold: float = 0.1,
    overlay_alpha: float = 0.6,
    base_asset: Optional[str] = None,
    overlay_asset: Optional[str] = None,
    max_pixels: int = 2400,
) -> Tuple[np.ndarray, str]:
    """Composite an overlay on a base product from two :class:`LoadedItem` objects.

    Loads each item from disk, renders the base colormap, applies the
    overlay where values exceed a threshold, then unloads both.

    Parameters
    ----------
    base_item, overlay_item : LoadedItem
        Data items for the base and overlay layers.
    base_cmap, overlay_cmap : str
        Colormap names.
    base_vmin, base_vmax : float
        Colour-scale limits for the base layer.
    overlay_threshold : float
        Only overlay pixels where the overlay value exceeds this.
    overlay_alpha : float
        Opacity of the overlay (0 = transparent, 1 = opaque).
    base_asset, overlay_asset : str, optional
        Asset keys.  When *None*, the first available key is used.

    Returns
    -------
    rgb : np.ndarray
        ``(H, W, 3)`` float32 composited array.
    label : str
        Human-readable label from the base item.
    """
    base_item.load()
    bk = base_asset if base_asset is not None else next(iter(base_item.data))
    barr = base_item.data[bk].values
    base_item.unload()

    if max_pixels and max(barr.shape[:2]) > max_pixels:
        barr = _downsample_2d(barr, max_pixels)

    bcmap = _get_cmap(base_cmap)
    bnorm = np.clip((barr - base_vmin) / (base_vmax - base_vmin + 1e-10), 0, 1)
    del barr
    rgb = bcmap(bnorm)[:, :, :3].astype(np.float32)
    del bnorm

    overlay_item.load()
    ok = overlay_asset if overlay_asset is not None else next(iter(overlay_item.data))
    oarr = overlay_item.data[ok].values
    overlay_item.unload()

    if max_pixels and max(oarr.shape[:2]) > max_pixels:
        oarr = _downsample_2d(oarr, max_pixels)

    mask = oarr > overlay_threshold
    if mask.any():
        ocmap = _get_cmap(overlay_cmap)
        onorm = np.clip(oarr, 0, 1)
        orgb = ocmap(onorm)[:, :, :3].astype(np.float32)
        del onorm
        rgb[mask] = (1 - overlay_alpha) * rgb[mask] + overlay_alpha * orgb[mask]
        del orgb
    del oarr, mask

    return rgb, base_item.label


def dual_panel_frames(
    left: xr.DataArray,
    right: xr.DataArray,
    left_cmap: str = "viridis",
    right_cmap: str = "viridis",
    left_vmin: Optional[float] = None,
    left_vmax: Optional[float] = None,
    right_vmin: Optional[float] = None,
    right_vmax: Optional[float] = None,
    left_label: str = "",
    right_label: str = "",
    time_dim: str = "time",
    step: int = 1,
    gap_px: int = 4,
) -> Tuple[List[np.ndarray], List[str]]:
    """Build side-by-side dual-panel RGB frames from two DataArrays.

    Parameters
    ----------
    left, right : xr.DataArray
        Arrays with matching time dimension length.
    left_cmap, right_cmap : str
        Colormaps for left and right panels.
    left_vmin, left_vmax, right_vmin, right_vmax : float | None
        Colour-scale limits.
    left_label, right_label : str
        Product labels (embedded in returned label strings).
    time_dim : str
        Name of the time dimension.
    step : int
        Subsample factor.
    gap_px : int
        Width in pixels of the white gap between panels.

    Returns
    -------
    frames : list[np.ndarray]
        Side-by-side RGB arrays ``(H, W_left + gap + W_right, 3)``.
    labels : list[str]
        Date labels per frame.
    """
    left_frames, left_labels = data_to_rgb_frames(
        left, left_cmap, left_vmin, left_vmax, time_dim, step,
    )
    right_frames, _ = data_to_rgb_frames(
        right, right_cmap, right_vmin, right_vmax, time_dim, step,
    )

    if len(left_frames) != len(right_frames):
        n = min(len(left_frames), len(right_frames))
        left_frames = left_frames[:n]
        right_frames = right_frames[:n]
        left_labels = left_labels[:n]

    frames: List[np.ndarray] = []
    labels: List[str] = []
    for lf, rf, lbl in zip(left_frames, right_frames, left_labels):
        h = max(lf.shape[0], rf.shape[0])
        # Pad shorter panel vertically if needed
        if lf.shape[0] < h:
            pad = np.ones((h - lf.shape[0], lf.shape[1], 3), dtype=np.float32)
            lf = np.vstack([lf, pad])
        if rf.shape[0] < h:
            pad = np.ones((h - rf.shape[0], rf.shape[1], 3), dtype=np.float32)
            rf = np.vstack([rf, pad])

        gap = np.ones((h, gap_px, 3), dtype=np.float32)
        combined = np.hstack([lf, gap, rf])
        frames.append(combined)

        tag = f"{left_label} | {right_label}" if left_label or right_label else ""
        labels.append(f"{lbl}  {tag}".strip())

    return frames, labels


def overlay_frames(
    base: xr.DataArray,
    overlay: xr.DataArray,
    base_cmap: str = "YlGn",
    overlay_cmap: str = "Reds",
    base_vmin: Optional[float] = None,
    base_vmax: Optional[float] = None,
    overlay_threshold: float = 0.1,
    overlay_alpha: float = 0.6,
    time_dim: str = "time",
    step: int = 1,
) -> Tuple[List[np.ndarray], List[str]]:
    """Create RGB frames with a semi-transparent overlay where values exceed a threshold.

    Useful for showing burnt area scars on top of NDVI, or anomaly
    regions on top of a baseline product.

    Parameters
    ----------
    base : xr.DataArray
        Background product (e.g. NDVI).
    overlay : xr.DataArray
        Overlay product (e.g. burnt area mask).
    overlay_threshold : float
        Only overlay pixels where the value exceeds this threshold.
    overlay_alpha : float
        Opacity of the overlay (0 = transparent, 1 = opaque).

    Returns
    -------
    frames : list[np.ndarray]
        Composited RGB arrays.
    labels : list[str]
        Date labels per frame.
    """
    base_frames, base_labels = data_to_rgb_frames(
        base, base_cmap, base_vmin, base_vmax, time_dim, step,
    )
    overlay_colormap = _get_cmap(overlay_cmap)

    times = overlay[time_dim].values
    frames: List[np.ndarray] = []
    labels: List[str] = []

    for i, (bf, lbl) in enumerate(zip(base_frames, base_labels)):
        t_idx = i * step
        if t_idx >= len(times):
            break
        ov_arr = overlay.isel({time_dim: t_idx}).values
        mask = ov_arr > overlay_threshold

        if mask.any():
            ov_norm = np.clip(ov_arr, 0, 1)
            ov_rgb = overlay_colormap(ov_norm)[:, :, :3].astype(np.float32)
            composited = bf.copy()
            composited[mask] = (
                (1 - overlay_alpha) * bf[mask] + overlay_alpha * ov_rgb[mask]
            )
            frames.append(composited)
        else:
            frames.append(bf)
        labels.append(lbl)

    return frames, labels


def product_cycle_frames(
    products: Dict[str, xr.DataArray],
    cmaps: Optional[Dict[str, str]] = None,
    vmins: Optional[Dict[str, float]] = None,
    vmaxs: Optional[Dict[str, float]] = None,
    time_index: int = 0,
    time_dim: str = "time",
) -> Tuple[List[np.ndarray], List[str]]:
    """Create one frame per product for a fixed time step.

    Useful for "product tour" animations that cycle through different
    views of the same region.

    Parameters
    ----------
    products : dict[str, xr.DataArray]
        Mapping ``product_name -> DataArray``.
    cmaps : dict[str, str] | None
        Per-product colormap overrides.
    vmins, vmaxs : dict[str, float] | None
        Per-product colour-scale limits.
    time_index : int
        Time step to visualize.

    Returns
    -------
    frames : list[np.ndarray]
        One RGB frame per product.
    labels : list[str]
        Product name per frame.
    """
    cmaps = cmaps or {}
    vmins = vmins or {}
    vmaxs = vmaxs or {}

    frames: List[np.ndarray] = []
    labels: List[str] = []

    for name, da in products.items():
        cmap = cmaps.get(name, "viridis")
        vmin = vmins.get(name)
        vmax = vmaxs.get(name)
        colormap = _get_cmap(cmap)

        arr = da.isel({time_dim: time_index}).values
        if vmin is None:
            vmin = float(np.nanpercentile(arr, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(arr, 98))

        norm = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
        rgb = colormap(norm)[:, :, :3].astype(np.float32)
        frames.append(rgb)
        labels.append(name)

    return frames, labels
