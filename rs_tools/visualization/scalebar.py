"""Scale bar for geo-referenced imagery plots.

Draws a horizontal bar annotated with ground distance (km) on a
matplotlib axes, given the pixel size in metres.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

_NICE_LENGTHS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

_LOC_MAP = {
    "upper right": 1,
    "upper left": 2,
    "lower left": 3,
    "lower right": 4,
}


def _auto_length_km(extent_pixels: float, pixel_size_m: float) -> float:
    """Pick a nice scale-bar length that is roughly ¼ of the image width."""
    target_km = extent_pixels * pixel_size_m / 1000.0 / 4.0
    return min(_NICE_LENGTHS, key=lambda x: abs(x - target_km))


def add_scalebar(
    ax: plt.Axes,
    pixel_size_m: float,
    length_km: Optional[float] = None,
    location: str = "lower right",
    color: str = "white",
    fontsize: int = 10,
    pad: float = 0.5,
    bar_height_frac: float = 0.02,
) -> AnchoredSizeBar:
    """Add a geodetic scale bar to a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (must already have image data plotted).
    pixel_size_m : float
        Ground distance per pixel in metres.
    length_km : float, optional
        Scale-bar length in kilometres.  Auto-determined if *None*.
    location : str
        One of ``"lower right"``, ``"lower left"``,
        ``"upper right"``, ``"upper left"``.
    color : str
        Bar and label colour.
    fontsize : int
        Label font size.
    pad : float
        Padding inside the frame.
    bar_height_frac : float
        Vertical thickness as a fraction of bar length.

    Returns
    -------
    AnchoredSizeBar
    """
    xlim = ax.get_xlim()
    extent_px = abs(xlim[1] - xlim[0])

    if length_km is None:
        length_km = _auto_length_km(extent_px, pixel_size_m)

    bar_length_px = length_km * 1000.0 / pixel_size_m

    label = f"{length_km:g} km"
    loc = _LOC_MAP.get(location, 4)

    scalebar = AnchoredSizeBar(
        ax.transData,
        bar_length_px,
        label,
        loc,
        pad=pad,
        color=color,
        frameon=True,
        size_vertical=max(bar_length_px * bar_height_frac, 2),
        fontproperties=fm.FontProperties(size=fontsize),
        sep=5,
        fill_bar=True,
    )
    scalebar.patch.set(facecolor="black", alpha=0.6, edgecolor="none")
    ax.add_artist(scalebar)
    return scalebar
