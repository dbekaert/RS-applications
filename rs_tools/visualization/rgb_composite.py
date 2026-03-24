"""RGB and multi-temporal composite imagery.

Functions to create true- or false-colour RGB composites from
multi-band or multi-temporal raster data, with optional histogram
stretching.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _normalize(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Normalise array to [0, 1] range."""
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))
    out = (arr - vmin) / (vmax - vmin + 1e-10)
    return np.clip(out, 0.0, 1.0)


def make_rgb(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Stack three 2-D arrays into an (H, W, 3) RGB image.

    Each channel is independently normalised to [0, 1] using a 2–98
    percentile stretch (or explicit *vmin*/*vmax*).

    Parameters
    ----------
    red, green, blue : np.ndarray
        2-D arrays of the same shape.
    vmin, vmax : float | None
        Fixed normalisation bounds applied to all channels.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` float array with values in [0, 1].
    """
    return np.dstack([
        _normalize(red, vmin, vmax),
        _normalize(green, vmin, vmax),
        _normalize(blue, vmin, vmax),
    ])


def multi_temporal_rgb(
    data: xr.DataArray,
    time_indices: Tuple[int, int, int],
    time_dim: str = "time",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Create an RGB composite from three time steps.

    Useful for showing seasonal change: assign different dates to the
    R, G, and B channels.

    Parameters
    ----------
    data : xr.DataArray
        3-D array ``(time, y, x)``.
    time_indices : tuple[int, int, int]
        Indices along *time_dim* mapped to (Red, Green, Blue).
    time_dim : str
        Name of the time dimension.
    vmin, vmax : float | None
        Normalisation bounds.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` RGB array.
    """
    r = data.isel({time_dim: time_indices[0]}).values
    g = data.isel({time_dim: time_indices[1]}).values
    b = data.isel({time_dim: time_indices[2]}).values
    return make_rgb(r, g, b, vmin=vmin, vmax=vmax)


def plot_rgb(
    rgb: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Display an RGB composite image.

    Parameters
    ----------
    rgb : np.ndarray
        ``(H, W, 3)`` RGB array (values 0–1).
    title : str | None
        Plot title.
    figsize : tuple
        Figure size (used only when *ax* is None).
    ax : matplotlib.axes.Axes | None
        Existing axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(rgb, origin="upper")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig
