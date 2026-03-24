"""Comparison slider plots.

Side-by-side or overlay comparison of two images with a draggable slider
divider — useful for before/after or multi-product comparisons.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Slider


def slider_comparison(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str = "Left",
    right_label: str = "Right",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Interactive slider comparison of two 2-D arrays.

    The slider controls a vertical divider: pixels to the left come from
    *left*, pixels to the right from *right*.

    Parameters
    ----------
    left, right : np.ndarray
        2-D arrays of the same shape.
    left_label, right_label : str
        Labels shown on the slider axis.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float | None
        Colour-scale limits.
    title : str | None
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if left.shape != right.shape:
        raise ValueError("left and right arrays must have the same shape.")

    rows, cols = left.shape
    composite = left.copy().astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(bottom=0.18)

    im = ax.imshow(composite, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    divider_line = ax.axvline(cols // 2, color="white", linewidth=2)

    ax_slider = fig.add_axes([0.15, 0.05, 0.70, 0.03])
    slider = Slider(
        ax_slider,
        f"{left_label}  ←→  {right_label}",
        0,
        cols - 1,
        valinit=cols // 2,
        valstep=1,
    )

    def _update(val: float) -> None:
        split = int(slider.val)
        composite[:, :split] = left[:, :split]
        composite[:, split:] = right[:, split:]
        im.set_data(composite)
        divider_line.set_xdata([split])
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    _update(cols // 2)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    return fig


def slider_comparison_xr(
    left: xr.DataArray,
    right: xr.DataArray,
    left_label: str = "Left",
    right_label: str = "Right",
    **kwargs,
) -> plt.Figure:
    """Convenience wrapper accepting ``xr.DataArray`` inputs."""
    return slider_comparison(
        left.values,
        right.values,
        left_label=left_label,
        right_label=right_label,
        **kwargs,
    )
