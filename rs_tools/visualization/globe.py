"""3-D globe inset visualization.

Renders a small orthographic globe on which the region of interest is
highlighted, useful as a location-context inset in figures.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from rs_tools.config import BoundingBox


def add_globe_inset(
    fig: plt.Figure,
    bbox: BoundingBox,
    position: Tuple[float, float, float, float] = (0.02, 0.60, 0.25, 0.25),
    land_color: str = "#e0e0e0",
    ocean_color: str = "#a8d5e2",
    highlight_color: str = "red",
    highlight_alpha: float = 0.5,
) -> plt.Axes:
    """Add a 3-D orthographic globe inset to an existing figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the inset to.
    bbox : BoundingBox
        Region of interest to highlight on the globe.
    position : tuple
        ``(left, bottom, width, height)`` in figure coordinates (0–1).
    land_color, ocean_color : str
        Colours for land masses and ocean.
    highlight_color : str
        Colour of the highlighted bounding-box rectangle.
    highlight_alpha : float
        Transparency of the highlight rectangle.

    Returns
    -------
    matplotlib.axes.Axes
        The inset axes (cartopy ``GeoAxes``).
    """
    center_lon = (bbox.west + bbox.east) / 2
    center_lat = (bbox.south + bbox.north) / 2

    proj = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    ax = fig.add_axes(position, projection=proj)
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="gray", linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor=ocean_color)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Draw the bounding box
    lons = [bbox.west, bbox.east, bbox.east, bbox.west, bbox.west]
    lats = [bbox.south, bbox.south, bbox.north, bbox.north, bbox.south]
    ax.fill(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        color=highlight_color,
        alpha=highlight_alpha,
    )
    ax.plot(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        color=highlight_color,
        linewidth=1.5,
    )

    return ax
