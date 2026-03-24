"""Map overlay utilities for annotating RTC composites.

Provides functions to fetch road networks from OpenStreetMap (via
the Overpass API) and project them onto raster pixel coordinates
for overlay on SAR imagery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def fetch_roads(
    bbox: Tuple[float, float, float, float],
    highway_types: str = "motorway|trunk|primary|secondary",
    timeout: int = 25,
) -> List[Dict[str, Any]]:
    """Fetch road geometries from OpenStreetMap via the Overpass API.

    Parameters
    ----------
    bbox : tuple
        ``(south, west, north, east)`` in WGS-84 degrees.
    highway_types : str
        Pipe-separated list of OSM highway types to fetch.
    timeout : int
        Overpass query timeout in seconds.

    Returns
    -------
    list[dict]
        List of OSM way elements, each with a ``geometry`` list of
        ``{"lat": ..., "lon": ...}`` nodes.
    """
    import requests

    south, west, north, east = bbox
    query = (
        f'[out:json][timeout:{timeout}];'
        f'way["highway"~"{highway_types}"]'
        f'({south},{west},{north},{east});'
        f'out geom;'
    )
    resp = requests.get(
        "https://overpass-api.de/api/interpreter",
        params={"data": query},
        timeout=timeout + 10,
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    logger.info("Fetched %d road segments from OSM", len(elements))
    return elements


def _lonlat_to_pixel(
    lon: float,
    lat: float,
    transform,
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32631",
) -> Tuple[float, float]:
    """Convert lon/lat to pixel row/col using a rasterio Affine transform."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return row, col


def overlay_roads(
    ax,
    roads: List[Dict[str, Any]],
    data_array,
    color: str = "#555555",
    linewidth: float = 0.5,
    alpha: float = 0.6,
) -> None:
    """Draw road network on a matplotlib axes aligned with a raster.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes displaying an ``imshow`` of the raster.
    roads : list[dict]
        OSM way elements from :func:`fetch_roads`.
    data_array : xarray.DataArray
        Geo-referenced DataArray (with ``.rio.crs`` and
        ``.rio.transform()``).
    color : str
        Road line colour.
    linewidth : float
        Road line width.
    alpha : float
        Road line opacity.
    """
    transform = data_array.rio.transform()
    dst_crs = str(data_array.rio.crs)

    for way in roads:
        geom = way.get("geometry", [])
        if len(geom) < 2:
            continue
        rows, cols = [], []
        for node in geom:
            r, c = _lonlat_to_pixel(
                node["lon"], node["lat"], transform,
                src_crs="EPSG:4326", dst_crs=dst_crs,
            )
            rows.append(r)
            cols.append(c)
        ax.plot(cols, rows, color=color, linewidth=linewidth,
                alpha=alpha, solid_capstyle="round")


def annotate_location(
    ax,
    lon: float,
    lat: float,
    label: str,
    data_array,
    color: str = "white",
    fontsize: int = 8,
    marker: str = "o",
    markersize: int = 4,
) -> None:
    """Place a labelled marker on a raster map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes showing an ``imshow`` of the raster.
    lon, lat : float
        Marker position in WGS-84.
    label : str
        Text label placed next to the marker.
    data_array : xarray.DataArray
        Geo-referenced DataArray for coordinate conversion.
    color : str
        Marker and text colour.
    fontsize : int
        Label font size.
    marker, markersize : str, int
        Matplotlib marker style and size.
    """
    transform = data_array.rio.transform()
    dst_crs = str(data_array.rio.crs)
    row, col = _lonlat_to_pixel(lon, lat, transform,
                                 src_crs="EPSG:4326", dst_crs=dst_crs)
    ax.plot(col, row, marker=marker, color=color, markersize=markersize,
            markeredgecolor="black", markeredgewidth=0.5)
    ax.annotate(
        label, (col, row),
        fontsize=fontsize, color=color, fontweight="bold",
        xytext=(6, -2), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6,
                  edgecolor="none"),
    )


def overlay_geojson(
    ax,
    geojson_path: str,
    data_array,
    *,
    category_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    default_color: str = "yellow",
    default_linewidth: float = 1.0,
    default_alpha: float = 0.7,
    label_field: Optional[str] = None,
    filter_categories: Optional[List[str]] = None,
) -> None:
    """Overlay GeoJSON features on a raster map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes displaying an ``imshow`` of the raster.
    geojson_path : str
        Path to a GeoJSON FeatureCollection file.
    data_array : xarray.DataArray
        Geo-referenced DataArray (with ``.rio.crs`` and
        ``.rio.transform()``).
    category_styles : dict, optional
        Mapping from category name → dict with ``color``, ``linewidth``,
        ``alpha``, ``linestyle`` keys (all optional).
    default_color, default_linewidth, default_alpha
        Fallback style for features without a matching category style.
    label_field : str, optional
        Property field to use as text labels (e.g. ``"name"``).
    filter_categories : list[str], optional
        If set, only features whose ``category`` property is in this
        list will be drawn.
    """
    import json as _json

    with open(geojson_path) as f:
        fc = _json.load(f)

    transform = data_array.rio.transform()
    dst_crs = str(data_array.rio.crs)
    category_styles = category_styles or {}

    # Track which categories have been added to the legend
    _legend_added: set = set()

    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        cat = props.get("category", "")

        if filter_categories and cat not in filter_categories:
            continue

        style = category_styles.get(cat, {})
        color = style.get("color", default_color)
        lw = style.get("linewidth", default_linewidth)
        alpha = style.get("alpha", default_alpha)
        ls = style.get("linestyle", "-")

        geom = feat.get("geometry", {})
        geom_type = geom.get("type", "")

        coord_rings = []
        if geom_type == "LineString":
            coord_rings.append(geom["coordinates"])
        elif geom_type == "Polygon":
            coord_rings.extend(geom["coordinates"])
        elif geom_type == "MultiLineString":
            coord_rings.extend(geom["coordinates"])
        else:
            continue

        # Legend label only for the first feature of each category
        legend_label = cat if cat not in _legend_added else None
        _legend_added.add(cat)

        for ring in coord_rings:
            rows, cols = [], []
            for lon, lat, *_ in ring:
                r, c = _lonlat_to_pixel(
                    lon, lat, transform,
                    src_crs="EPSG:4326", dst_crs=dst_crs,
                )
                rows.append(r)
                cols.append(c)
            ax.plot(cols, rows, color=color, linewidth=lw,
                    alpha=alpha, linestyle=ls,
                    solid_capstyle="round", label=legend_label)
            legend_label = None  # Only label the first ring

        # Optionally place a text label at the feature centroid
        if label_field and props.get(label_field):
            first_ring = coord_rings[0]
            mid = first_ring[len(first_ring) // 2]
            r, c = _lonlat_to_pixel(
                mid[0], mid[1], transform,
                src_crs="EPSG:4326", dst_crs=dst_crs,
            )
            ax.annotate(
                props[label_field], (c, r),
                fontsize=6, color=color, alpha=0.9,
                xytext=(4, 2), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="black", alpha=0.5, edgecolor="none"),
            )
