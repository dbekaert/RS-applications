"""Time-series visualization with interactive slider.

Creates animated or interactive time-series figures with a date slider
on top, suitable for exploring temporal evolution of remote-sensing
products.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Slider


def plot_timeseries_slider(
    data: xr.DataArray,
    time_dim: str = "time",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Display a 2-D spatial field with a slider to step through time.

    Parameters
    ----------
    data : xr.DataArray
        3-D array with dimensions ``(time, y, x)`` (or equivalent).
    time_dim : str
        Name of the time dimension.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float | None
        Colour-scale limits.  Computed from data if *None*.
    title : str | None
        Figure super-title.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = data[time_dim].values
    n_steps = len(times)

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(
        data.isel({time_dim: 0}).values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_slider = fig.add_axes([0.15, 0.05, 0.70, 0.03])
    slider = Slider(
        ax_slider,
        "Time",
        0,
        n_steps - 1,
        valinit=0,
        valstep=1,
    )

    def _update(val: float) -> None:
        idx = int(slider.val)
        im.set_data(data.isel({time_dim: idx}).values)
        time_label = str(np.datetime_as_string(times[idx], unit="D"))
        ax.set_title(time_label)
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    _update(0)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    return fig


def plot_timeseries_line(
    data: xr.DataArray,
    time_dim: str = "time",
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot one or more 1-D time-series as line charts.

    Parameters
    ----------
    data : xr.DataArray
        1-D or 2-D array.  If 2-D the non-time dimension is iterated to
        produce multiple lines.
    time_dim : str
        Name of the time dimension.
    labels : list[str] | None
        Legend labels.
    title : str | None
        Plot title.
    ylabel : str | None
        Y-axis label.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    times = data[time_dim].values

    if data.ndim == 1:
        ax.plot(times, data.values, label=labels[0] if labels else None)
    else:
        other_dim = [d for d in data.dims if d != time_dim][0]
        for i, coord in enumerate(data[other_dim].values):
            label = labels[i] if labels and i < len(labels) else str(coord)
            ax.plot(times, data.sel({other_dim: coord}).values, label=label)

    ax.set_xlabel("Date")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if labels:
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
