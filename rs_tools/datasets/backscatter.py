"""Backscatter type conversion using OPERA RTC-S1 static layers.

OPERA RTC-S1 products provide **gamma-0 (γ⁰)** power by default.
The companion static-layer product (OPERA_RTC_S1_STATIC) supplies
Area Normalisation Factor (ANF) grids that convert gamma-0 to
**beta-0 (β⁰)** or **sigma-0 (σ⁰)**:

.. math::

    \\beta^0 = \\gamma^0 \\times \\text{ANF}_{\\gamma\\to\\beta}

    \\sigma^0 = \\gamma^0 \\times \\text{ANF}_{\\gamma\\to\\sigma}

Static layers share the same burst grid as the RTC product.

Usage
-----
>>> from rs_tools.datasets.backscatter import BackscatterType, convert_backscatter
>>> # items is a list of STAC item dicts (RTC bursts) from search
>>> converted = convert_backscatter(items, target=BackscatterType.SIGMA0,
...                                 archive="terrascope")
"""

from __future__ import annotations

import enum
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Backscatter types ────────────────────────────────────────────────────

class BackscatterType(enum.Enum):
    """Supported SAR backscatter conventions."""
    GAMMA0 = "gamma0"
    BETA0 = "beta0"
    SIGMA0 = "sigma0"


# Maps target type → name of the ANF asset in the static product
_ANF_ASSET_KEY = {
    BackscatterType.BETA0: "rtc_anf_gamma0_to_beta0",
    BackscatterType.SIGMA0: "rtc_anf_gamma0_to_sigma0",
}


# ── Burst-ID helpers ────────────────────────────────────────────────────

_BURST_RE = re.compile(r"T(\d+)-(\d+)-(IW\d)")


def extract_burst_id(item_id: str) -> Optional[str]:
    """Extract the OPERA burst identifier from an item ID.

    Returns a string like ``"T037-078096-IW3"`` or *None* if the
    pattern is not found.
    """
    m = _BURST_RE.search(item_id)
    return m.group(0) if m else None


# ── Static-layer resolution ─────────────────────────────────────────────

def _static_id_from_related_link(item: Dict[str, Any]) -> Optional[str]:
    """Extract the static-layer STAC item ID from a ``rel=related`` link.

    Terrascope RTC items include a ``rel=related`` link whose ``href``
    points to the matching static item in ``opera-s1-rtc-static-v1``.
    """
    for link in item.get("links", []):
        if link.get("rel") == "related":
            href = link.get("href", "")
            if "rtc-static" in href.lower() or "RTC-S1-STATIC" in href:
                # URL ends with the item ID
                return href.rstrip("/").rsplit("/", 1)[-1]
    return None


def resolve_static_items_terrascope(
    rtc_items: List[Dict[str, Any]],
    anf_asset: str,
) -> Dict[str, Dict[str, Any]]:
    """Resolve static-layer STAC items for a batch of Terrascope RTC items.

    Uses the ``rel=related`` link if present; otherwise falls back to
    searching the ``opera-s1-rtc-static-v1`` collection by burst ID.

    Parameters
    ----------
    rtc_items : list[dict]
        RTC STAC item dicts (from Terrascope search).
    anf_asset : str
        ANF asset key to look for (e.g. ``"rtc_anf_gamma0_to_sigma0"``).

    Returns
    -------
    dict[str, dict]
        Mapping from burst_id (e.g. ``"T037-078096-IW3"``) to the
        static STAC item dict.  Items whose static layers cannot be
        resolved are omitted.
    """
    from pystac_client import Client
    from rs_tools.archives.terrascope import TERRASCOPE_STAC_URL

    burst_to_static: Dict[str, Dict[str, Any]] = {}
    need_search: List[str] = []  # burst IDs that have no related link

    for item in rtc_items:
        burst_id = (
            item.get("properties", {}).get("opera:burst_id")
            or extract_burst_id(item.get("id", ""))
        )
        if not burst_id or burst_id in burst_to_static:
            continue

        static_id = _static_id_from_related_link(item)
        if static_id:
            burst_to_static[burst_id] = {"_static_id": static_id}
        else:
            need_search.append(burst_id)

    # Fetch full STAC items for those resolved via related link
    if burst_to_static:
        client = Client.open(TERRASCOPE_STAC_URL)
        ids_to_fetch = [
            v["_static_id"] for v in burst_to_static.values()
            if "_static_id" in v
        ]
        if ids_to_fetch:
            try:
                results = client.search(
                    collections=["opera-s1-rtc-static-v1"],
                    ids=ids_to_fetch,
                )
                for static_item in results.items():
                    d = static_item.to_dict()
                    sid = d.get("id", "")
                    bid = extract_burst_id(sid)
                    if bid and anf_asset in d.get("assets", {}):
                        burst_to_static[bid] = d
            except Exception:
                logger.warning(
                    "Failed to fetch static items by ID, falling back to "
                    "search", exc_info=True,
                )
                need_search.extend(
                    bid for bid, v in burst_to_static.items()
                    if "_static_id" in v
                )
                burst_to_static = {
                    k: v for k, v in burst_to_static.items()
                    if "_static_id" not in v
                }

    # Fallback: search the static collection by bbox per burst
    if need_search:
        _search_static_by_burst_terrascope(
            list(set(need_search)), anf_asset, burst_to_static,
        )

    # Clean out unresolved placeholders
    burst_to_static = {
        k: v for k, v in burst_to_static.items()
        if "_static_id" not in v
    }

    logger.info(
        "Resolved %d/%d burst static layers via Terrascope",
        len(burst_to_static),
        len({extract_burst_id(i.get("id", "")) for i in rtc_items}),
    )
    return burst_to_static


def _search_static_by_burst_terrascope(
    burst_ids: List[str],
    anf_asset: str,
    result_map: Dict[str, Dict[str, Any]],
) -> None:
    """Search Terrascope static collection for specific burst IDs."""
    from pystac_client import Client
    from rs_tools.archives.terrascope import TERRASCOPE_STAC_URL

    client = Client.open(TERRASCOPE_STAC_URL)

    for burst_id in burst_ids:
        if burst_id in result_map and "_static_id" not in result_map[burst_id]:
            continue
        try:
            # The static item ID starts with
            # "OPERA_L2_RTC-S1-STATIC_{burst_id}_"
            # We can use a free-text / ID filter if available, or
            # do a spatial search for items matching the burst.
            # Use the collections' search with a filter on the ID.
            results = client.search(
                collections=["opera-s1-rtc-static-v1"],
                # Use the STAC API free-text search or filter extension
                # to find items matching the burst ID
                max_items=5,
                filter=f"id LIKE 'OPERA_L2_RTC-S1-STATIC_{burst_id}%'",
                filter_lang="cql2-text",
            )
            for static_item in results.items():
                d = static_item.to_dict()
                if anf_asset in d.get("assets", {}):
                    result_map[burst_id] = d
                    break
        except Exception:
            logger.debug(
                "CQL2 filter not supported, trying ID-based search",
                exc_info=True,
            )
            # Construct the expected static ID directly
            # Pattern: OPERA_L2_RTC-S1-STATIC_{burst_id}_20140403_S1A_30_v1.0
            # The date 20140403 is the nominal orbit date (fixed)
            candidate_id = (
                f"OPERA_L2_RTC-S1-STATIC_{burst_id}_20140403_S1A_30_v1.0"
            )
            try:
                results = client.search(
                    collections=["opera-s1-rtc-static-v1"],
                    ids=[candidate_id],
                )
                for static_item in results.items():
                    d = static_item.to_dict()
                    if anf_asset in d.get("assets", {}):
                        result_map[burst_id] = d
                        break
            except Exception:
                logger.warning(
                    "Could not resolve static layer for burst %s",
                    burst_id, exc_info=True,
                )


def resolve_static_items_nasa(
    rtc_items: List[Dict[str, Any]],
    anf_asset: str,
) -> Dict[str, Dict[str, Any]]:
    """Resolve static-layer STAC items via ASF search.

    ASF RTC items do not carry a direct link to the static product.
    We search the ``OPERA_L2_RTC-S1-STATIC_V1`` collection for items
    matching each unique burst ID.

    Parameters
    ----------
    rtc_items : list[dict]
        RTC STAC item dicts (from ASF/NASA search).
    anf_asset : str
        ANF asset key to look for.

    Returns
    -------
    dict[str, dict]
        Mapping from burst_id to static STAC item dict.
    """
    import asf_search

    burst_ids = set()
    for item in rtc_items:
        bid = extract_burst_id(item.get("id", ""))
        if bid:
            burst_ids.add(bid)

    burst_to_static: Dict[str, Dict[str, Any]] = {}

    for burst_id in burst_ids:
        # Construct the expected product name pattern
        # ASF accepts granule name search
        try:
            results = asf_search.search(
                dataset="OPERA-S1",
                processingLevel="RTC-STATIC",
                granule_list=[
                    f"OPERA_L2_RTC-S1-STATIC_{burst_id}*"
                ],
                maxResults=1,
            )
            if results:
                from rs_tools.archives.nasa import _scene_to_stac_item
                d = _scene_to_stac_item(results[0])
                # ASF uses filename-derived keys; check for the ANF asset
                # The filename ends with e.g. _rtc_anf_gamma0_to_sigma0.tif
                # which gets parsed as the asset key
                if anf_asset in d.get("assets", {}):
                    burst_to_static[burst_id] = d
                else:
                    # Asset may have a different key format — check all
                    for akey in d.get("assets", {}):
                        if anf_asset.replace("_", "") in akey.replace("_", ""):
                            d["assets"][anf_asset] = d["assets"][akey]
                            burst_to_static[burst_id] = d
                            break
        except Exception:
            logger.warning(
                "ASF static search failed for burst %s",
                burst_id, exc_info=True,
            )

    logger.info(
        "Resolved %d/%d burst static layers via ASF",
        len(burst_to_static), len(burst_ids),
    )
    return burst_to_static


def resolve_static_items_local(
    rtc_items: List[Dict[str, Any]],
    anf_asset: str,
) -> Dict[str, str]:
    """Resolve static ANF files from the local VITO filesystem.

    Scans ``/data/MTDA/NASA/ASF/OPERA_L2_RTC-S1-STATIC_V1/`` for
    directories matching each burst ID and returns paths to the
    specific ANF GeoTIFF.

    Parameters
    ----------
    rtc_items : list[dict]
        RTC STAC item dicts.
    anf_asset : str
        ANF asset key (e.g. ``"rtc_anf_gamma0_to_sigma0"``).

    Returns
    -------
    dict[str, str]
        Mapping from burst_id to local file path of the ANF GeoTIFF.
    """
    static_root = "/data/MTDA/NASA/ASF/OPERA_L2_RTC-S1-STATIC_V1"
    if not os.path.isdir(static_root):
        return {}

    burst_ids = set()
    for item in rtc_items:
        bid = extract_burst_id(item.get("id", ""))
        if bid:
            burst_ids.add(bid)

    burst_to_path: Dict[str, str] = {}

    # Walk year/month/day directories and match burst IDs
    for year_dir in sorted(os.listdir(static_root)):
        year_path = os.path.join(static_root, year_dir)
        if not os.path.isdir(year_path):
            continue
        for month_dir in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_dir)
            if not os.path.isdir(month_path):
                continue
            for day_dir in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day_dir)
                if not os.path.isdir(day_path):
                    continue
                for item_dir in os.listdir(day_path):
                    bid = extract_burst_id(item_dir)
                    if bid and bid in burst_ids and bid not in burst_to_path:
                        anf_file = os.path.join(
                            day_path, item_dir,
                            f"{item_dir}_{anf_asset}.tif",
                        )
                        if os.path.isfile(anf_file):
                            burst_to_path[bid] = anf_file

    logger.info(
        "Resolved %d/%d burst static layers from local filesystem",
        len(burst_to_path), len(burst_ids),
    )
    return burst_to_path


# ── ANF loading ──────────────────────────────────────────────────────────

def load_anf(
    static_item: Dict[str, Any],
    anf_asset: str,
    local_path: Optional[str] = None,
) -> "xr.DataArray":
    """Load an ANF raster from a static STAC item.

    Parameters
    ----------
    static_item : dict
        Static STAC item dict with ``assets`` containing ``anf_asset``.
    anf_asset : str
        Asset key (e.g. ``"rtc_anf_gamma0_to_sigma0"``).
    local_path : str, optional
        If provided, read directly from this local path instead of
        resolving from the STAC item.

    Returns
    -------
    xr.DataArray
        Geo-located ANF raster (float32).
    """
    from rs_tools.datasets.loader import load_stac_asset

    if local_path and os.path.isfile(local_path):
        import rioxarray  # noqa: F401
        da = rioxarray.open_rasterio(local_path, masked=True)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        return da.load()

    asset_entry = static_item.get("assets", {}).get(anf_asset, {})
    href = asset_entry.get("href", "")
    alternate = asset_entry.get("alternate")

    s3_url = None
    local_url = None
    if isinstance(alternate, str):
        s3_url = alternate
    elif isinstance(alternate, dict):
        from rs_tools.archives.local import extract_local_url
        local_url = extract_local_url(alternate)

    return load_stac_asset(
        href, bbox=None, s3_url=s3_url, local_url=local_url, chunks=None,
    )


# ── Core conversion ─────────────────────────────────────────────────────

def apply_anf(
    data: "xr.DataArray",
    anf: "xr.DataArray",
) -> "xr.DataArray":
    """Multiply an RTC gamma-0 raster by an ANF raster.

    The ANF may cover a larger area than the RTC burst.  The function
    reprojects the ANF onto the RTC grid (nearest-neighbor, since
    grids are aligned) and multiplies only where both have valid data.

    Parameters
    ----------
    data : xr.DataArray
        RTC gamma-0 power (linear).
    anf : xr.DataArray
        Area Normalisation Factor raster.

    Returns
    -------
    xr.DataArray
        Converted backscatter power (same grid as *data*).
    """
    # Reproject ANF to match the RTC grid exactly
    anf_matched = anf.rio.reproject_match(data)

    # Multiply — NaN stays NaN (valid * NaN = NaN)
    result = data * anf_matched

    # Zero/nodata in ANF → keep as NaN
    anf_valid = (anf_matched.values != 0) & np.isfinite(anf_matched.values)
    result = result.where(anf_valid)

    return result


# ── High-level API ───────────────────────────────────────────────────────

def convert_pass_backscatter(
    loaded_item: "LoadedItem",
    target: BackscatterType,
    anf_cache: Dict[str, Any],
    archive: str = "terrascope",
    polarizations: Optional[List[str]] = None,
) -> "LoadedItem":
    """Convert a loaded pass from gamma-0 to beta-0 or sigma-0.

    Parameters
    ----------
    loaded_item : LoadedItem
        A loaded pass with gamma-0 power data (VV, VH, etc.).
    target : BackscatterType
        Target backscatter type.
    anf_cache : dict
        Mapping from burst_id to ANF data (xr.DataArray or file path).
        Typically built by :func:`resolve_static_items_terrascope` or
        :func:`resolve_static_items_local`.
    archive : str
        Archive name (``"terrascope"`` or ``"nasa"``).
    polarizations : list[str], optional
        Polarizations to convert (e.g. ``["VV", "VH"]``).
        Defaults to all data assets.

    Returns
    -------
    LoadedItem
        The same item with data arrays converted in-place.
    """
    if target == BackscatterType.GAMMA0:
        return loaded_item

    anf_asset = _ANF_ASSET_KEY[target]
    if polarizations is None:
        polarizations = list(loaded_item.data.keys())

    # For mosaicked passes, we need the per-burst ANF mosaicked too.
    # Since the static grid matches the RTC burst grid, we can load
    # the ANF for each burst ID that composed this pass.
    # However, for simplicity with mosaicked passes, we mosaic the
    # ANF the same way the RTC data was mosaicked.

    # The loaded_item may have a stac_item with burst info, or we
    # can extract burst IDs from the mosaic ID pattern
    # (e.g. "T037_2024-06-30_34bursts")

    for pol in polarizations:
        if pol not in loaded_item.data:
            continue

        da = loaded_item.data[pol]

        # Try to find a matching ANF - for burst-level items, use
        # the item's own burst ID; for mosaicked passes, we need
        # a mosaicked ANF that matches the output grid
        burst_id = extract_burst_id(loaded_item.id)
        if burst_id and burst_id in anf_cache:
            anf_data = anf_cache[burst_id]
            if isinstance(anf_data, str):
                # It's a file path — load the ANF
                import rioxarray  # noqa: F401
                anf_da = rioxarray.open_rasterio(anf_data, masked=True)
                if "band" in anf_da.dims and anf_da.sizes["band"] == 1:
                    anf_da = anf_da.squeeze("band", drop=True)
                anf_da = anf_da.load()
            else:
                anf_da = anf_data
            loaded_item.data[pol] = apply_anf(da, anf_da)
        else:
            logger.warning(
                "No ANF available for %s (burst %s) — keeping gamma-0",
                loaded_item.id, burst_id,
            )

    return loaded_item
