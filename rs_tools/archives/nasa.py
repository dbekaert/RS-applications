"""Connector for NASA/ASF data access using asf_search.

Uses the ``asf_search`` library to query ASF's CMR-based catalogue and
returns items with HTTPS download URLs from ``datapool.asf.alaska.edu``.
When running on AWS, S3 URLs (``s3://asf-cumulus-prod-*``) are also
available — the loader can switch to ``/vsis3/`` in that case.

Authentication requires NASA Earthdata credentials in ``~/.netrc``::

    machine urs.earthdata.nasa.gov
        login <username>
        password <password>

GDAL streams COGs via ``/vsicurl/`` using cookie-based auth against
Earthdata's OAuth redirect chain (same approach as ARIA-tools).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from rs_tools.archives.base import BaseArchive
from rs_tools.config import SearchConfig

logger = logging.getLogger(__name__)

# ASF dataset / processingLevel constants
_DATASET_MAP: Dict[str, Dict[str, Any]] = {
    "OPERA_L2_RTC-S1_V1": {
        "dataset": "OPERA-S1",
        "processingLevel": "RTC",
    },
    "OPERA_L2_RTC-S1_V1_1": {
        "dataset": "OPERA-S1",
        "processingLevel": "RTC",
    },
    "OPERA_L2_RTC-S1-STATIC_V1": {
        "dataset": "OPERA-S1",
        "processingLevel": "RTC-STATIC",
    },
    "ARIA_S1_GUNW": {
        "dataset": "ARIA S1 GUNW",
        "processingLevel": "GUNW_STD",
    },
    "S1_SLC_BURST": {
        "dataset": "SLC-BURST",
    },
}


def _bbox_to_wkt(bbox: List[float]) -> str:
    """Convert [west, south, east, north] to a WKT POLYGON."""
    w, s, e, n = bbox
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _scene_to_stac_item(scene) -> Dict[str, Any]:
    """Convert an asf_search result into a STAC-like item dictionary.

    Builds ``assets`` with keys like ``VV``, ``VH``, ``mask`` by
    matching filenames from the ``url`` and ``additionalUrls`` properties.
    """
    geojson = scene.geojson()
    props = geojson.get("properties", {})
    scene_name = props.get("sceneName", "")

    # Collect all HTTPS download URLs
    all_urls: List[str] = []
    main_url = props.get("url", "")
    if main_url:
        all_urls.append(main_url)
    for u in props.get("additionalUrls", []):
        if u not in all_urls:
            all_urls.append(u)

    # Collect S3 URLs (product bucket only, not browse)
    s3_urls: Dict[str, str] = {}
    for u in props.get("s3Urls", []):
        if "opera-products" in u and u.endswith(".tif"):
            s3_urls[os.path.basename(u)] = u

    # Build assets by matching filename suffixes
    assets: Dict[str, Dict[str, str]] = {}
    for url in all_urls:
        fname = os.path.basename(url)
        if not fname.endswith(".tif"):
            continue
        # Detect asset type from filename suffix
        asset_key = None
        if fname.endswith("_VV.tif"):
            asset_key = "VV"
        elif fname.endswith("_VH.tif"):
            asset_key = "VH"
        elif fname.endswith("_mask.tif"):
            asset_key = "mask"
        elif fname.endswith("_HH.tif"):
            asset_key = "HH"
        elif fname.endswith("_HV.tif"):
            asset_key = "HV"
        # SLC-BURST: polarization in middle of filename
        elif "_VV_" in fname:
            asset_key = "VV"
        elif "_VH_" in fname:
            asset_key = "VH"
        elif "_HH_" in fname:
            asset_key = "HH"
        elif "_HV_" in fname:
            asset_key = "HV"
        else:
            # Generic — use filename without scene prefix as key
            asset_key = fname.replace(scene_name + "_", "").replace(".tif", "")
            if not asset_key:
                asset_key = "data"

        asset_entry: Dict[str, str] = {
            "href": url,
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
        }
        # Attach S3 URL if available
        if fname in s3_urls:
            asset_entry["alternate"] = s3_urls[fname]
        assets[asset_key] = asset_entry

    # Map ASF properties to STAC-like properties
    start_time = props.get("startTime", "")
    stac_props: Dict[str, Any] = {
        "datetime": start_time,
        "start_datetime": start_time,
        "end_datetime": props.get("stopTime", ""),
        "platform": props.get("platform", ""),
        "constellation": "sentinel-1",
        "sat:orbit_state": (props.get("flightDirection") or "").lower()
            or None,
        "sat:absolute_orbit": props.get("orbit"),
    }

    return {
        "type": "Feature",
        "id": scene_name,
        "geometry": geojson.get("geometry"),
        "properties": stac_props,
        "assets": assets,
    }


def configure_gdal_nasa() -> None:
    """Configure GDAL environment for NASA data access.

    Sets up efficient COG streaming via ``/vsicurl/`` with cookie-based
    auth for NASA Earthdata's OAuth redirect chain, and enables block
    caching for performance.
    """
    settings = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt",
        # Cookie-based auth for Earthdata OAuth redirects
        "GDAL_HTTP_COOKIEFILE": os.path.join(
            os.path.expanduser("~"), ".cache", "rs_tools", "cookies.txt"
        ),
        "GDAL_HTTP_COOKIEJAR": os.path.join(
            os.path.expanduser("~"), ".cache", "rs_tools", "cookies.txt"
        ),
        # Block caching for vsicurl
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "67108864",  # 64 MB
        "GDAL_CACHEMAX": "256",  # MB
    }
    cookie_dir = os.path.dirname(settings["GDAL_HTTP_COOKIEFILE"])
    os.makedirs(cookie_dir, exist_ok=True)

    for key, value in settings.items():
        os.environ.setdefault(key, value)
    logger.info("GDAL environment configured for NASA data")


# Module-level authenticated session (populated on first use)
_nasa_session: Optional["requests.Session"] = None


def _get_nasa_session() -> "requests.Session":
    """Return a ``requests.Session`` authenticated via ``~/.netrc``.

    The session follows NASA Earthdata's OAuth redirect chain
    automatically using the credentials stored in ``~/.netrc``
    for ``urs.earthdata.nasa.gov``.
    """
    global _nasa_session  # noqa: PLW0603
    if _nasa_session is not None:
        return _nasa_session

    import netrc as _netrc
    import requests
    from pathlib import Path

    session = requests.Session()

    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            info = _netrc.netrc(str(netrc_path))
            auth = info.authenticators("urs.earthdata.nasa.gov")
            if auth:
                session.auth = (auth[0], auth[2])
                logger.info("NASA Earthdata session using .netrc credentials")
        except _netrc.NetrcParseError:
            logger.warning("Failed to parse ~/.netrc for NASA credentials")

    _nasa_session = session
    return session


def resolve_nasa_href(
    href: str,
    s3_url: Optional[str] = None,
) -> str:
    """Resolve an ASF asset URL to a GDAL-ready virtual path.

    On AWS with an S3 URL available, returns ``/vsis3/...`` for direct
    S3 access (fastest).  Otherwise returns ``/vsicurl/<href>``
    for streaming via HTTPS with cookie-based auth — no full download
    needed.

    Parameters
    ----------
    href : str
        HTTPS URL on ``datapool.asf.alaska.edu``.
    s3_url : str, optional
        Corresponding ``s3://`` URI from the STAC asset ``alternate`` key.

    Returns
    -------
    str
        GDAL virtual-filesystem path.
    """
    from rs_tools.archives.s3 import resolve_href
    return resolve_href(href, s3_url=s3_url)


def download_nasa_asset(href: str, cache_dir: Optional[str] = None) -> str:
    """Download an ASF HTTPS asset to a local cache, returning the path.

    .. deprecated::
        Prefer :func:`resolve_nasa_href` which streams COGs via
        ``/vsicurl/`` or ``/vsis3/`` without downloading the full file.

    Parameters
    ----------
    href : str
        HTTPS URL on ``datapool.asf.alaska.edu``.
    cache_dir : str, optional
        Directory for cached files.  Defaults to
        ``~/.cache/rs_tools/nasa/`` (override with env var
        ``RS_TOOLS_CACHE_DIR``).

    Returns
    -------
    str
        Path to the local file.
    """
    if cache_dir is None:
        cache_dir = os.environ.get(
            "RS_TOOLS_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "rs_tools", "nasa"),
        )
    os.makedirs(cache_dir, exist_ok=True)

    fname = os.path.basename(href)
    local_path = os.path.join(cache_dir, fname)

    if os.path.exists(local_path):
        logger.debug("Cache hit: %s", local_path)
        return local_path

    session = _get_nasa_session()
    resp = session.get(href, stream=True, timeout=60)
    resp.raise_for_status()

    with open(local_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)

    logger.info("Downloaded %s -> %s", fname, local_path)
    return local_path


def _merge_slc_burst_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge SLC-BURST items that differ only by polarization.

    ASF returns separate scenes for each polarization channel (VV, VH)
    of the same burst.  This function merges them into single STAC-like
    items with both polarizations as separate assets.
    """
    from collections import OrderedDict

    groups: Dict[str, Dict[str, Any]] = OrderedDict()
    for item in items:
        item_id = item.get("id", "")
        # Build a merge key by neutralising the polarization token
        merge_key = item_id
        for pol in ("_VV_", "_VH_", "_HH_", "_HV_"):
            merge_key = merge_key.replace(pol, "_XX_")

        if merge_key in groups:
            groups[merge_key]["assets"].update(item.get("assets", {}))
        else:
            groups[merge_key] = {
                **item,
                "assets": dict(item.get("assets", {})),
            }

    merged = list(groups.values())
    n_merged = len(items) - len(merged)
    if n_merged > 0:
        logger.info(
            "Merged %d SLC-BURST polarisation pairs → %d items",
            len(items), len(merged),
        )
    return merged


class NASAArchive(BaseArchive):
    """Interface to NASA ASF archive via ``asf_search``.

    Supports OPERA RTC, ARIA GUNW, and Sentinel-1 SLC-BURST products.
    """

    name = "nasa"

    def list_collections(self) -> List[str]:
        return list(_DATASET_MAP.keys())

    def search(self, config: SearchConfig) -> List[Dict[str, Any]]:
        """Search ASF for products matching the given criteria.

        Parameters
        ----------
        config : SearchConfig
            Search parameters (bbox, dates, collections, limit).

        Returns
        -------
        list[dict]
            STAC-like item dictionaries with ``assets`` containing
            HTTPS download URLs.
        """
        import asf_search

        wkt = _bbox_to_wkt(config.bbox.as_list())

        # Resolve dataset/processingLevel from collection name
        search_kwargs: Dict[str, Any] = {
            "intersectsWith": wkt,
            "start": f"{config.start_date}T00:00:00Z",
            "end": f"{config.end_date}T23:59:59Z",
            "maxResults": config.limit,
        }

        if config.collections:
            coll = config.collections[0]
            mapping = _DATASET_MAP.get(coll, {})
            if mapping:
                search_kwargs["dataset"] = mapping["dataset"]
                if "processingLevel" in mapping:
                    search_kwargs["processingLevel"] = mapping["processingLevel"]
            else:
                search_kwargs["dataset"] = coll

        results = asf_search.search(**search_kwargs)
        logger.info("ASF search returned %d scenes.", len(results))

        items = [_scene_to_stac_item(r) for r in results]

        # SLC-BURST: merge items that differ only by polarization
        if config.collections and config.collections[0] == "S1_SLC_BURST":
            items = _merge_slc_burst_items(items)

        return items
