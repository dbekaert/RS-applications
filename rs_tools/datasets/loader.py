"""Data loading for known datasets.

Searches STAC catalogs and loads COG assets as geo-located xarray
DataArrays.  Requires ``rioxarray`` (optional dependency).
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

from rs_tools.config import BoundingBox, SearchConfig
from rs_tools.datasets.catalog import get as get_dataset
from rs_tools.search import search_archive

logger = logging.getLogger(__name__)

# Default assets to load per dataset short name
_DEFAULT_ASSETS: Dict[str, List[str]] = {
    "OPERA_RTC_S1": ["VV", "VH"],
    "OPERA_RTC_S1_STATIC": ["mask"],
}

_SENSOR_NAMES = {
    "S1A": "Sentinel-1A",
    "S1B": "Sentinel-1B",
}


@dataclass
class LoadedItem:
    """A loaded STAC item with geo-located data and metadata.

    When *output_dir* is used in :func:`load_items`, data arrays are
    saved to GeoTIFF files on disk and freed from memory.  Use
    :meth:`load` / :meth:`unload` to page data in and out.
    """

    id: str
    datetime: datetime
    platform: str
    orbit_direction: Optional[str] = None  # "ascending" or "descending"
    data: Dict[str, Any] = field(default_factory=dict)
    crs: Optional[str] = None
    pixel_size_m: Optional[float] = None
    stac_item: Optional[Dict[str, Any]] = field(default=None, repr=False)
    pass_dir: Optional[str] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """Human-readable label: 'Sentinel-1A | ASC | 2024-06-30 17:41 UTC'."""
        parts = [self.platform]
        if self.orbit_direction:
            parts.append(self.orbit_direction[:3].upper())
        parts.append(self.datetime.strftime("%Y-%m-%d %H:%M UTC"))
        return " | ".join(parts)

    # ── Disk persistence ──────────────────────────────────────────────

    def _pass_key(self) -> str:
        """Build a chronological directory name for this pass."""
        dt_str = self.datetime.strftime("%Y%m%d_%H%M")
        orb = (self.orbit_direction or "UNK")[:3].upper()
        # Extract track from the mosaic id (e.g. T110_2022-01-16_3bursts)
        track_str = ""
        if self.id and self.id.startswith("T"):
            track_str = self.id.split("_")[0]  # "T110"
        else:
            parsed = parse_opera_rtc_id(self.id)
            if parsed.get("track") is not None:
                track_str = f"T{parsed['track']:03d}"
        parts = [dt_str, orb]
        if track_str:
            parts.append(track_str)
        return "_".join(parts)

    def save(self, output_dir: str) -> str:
        """Write data arrays as GeoTIFF + metadata JSON to *output_dir*.

        Returns the per-pass directory path.  After saving, call
        :meth:`unload` to free memory while keeping *pass_dir* for
        later :meth:`load`.
        """
        pass_key = self._pass_key()
        pass_dir = os.path.join(output_dir, "passes", pass_key)
        os.makedirs(pass_dir, exist_ok=True)

        for asset_name, da in self.data.items():
            tif_path = os.path.join(pass_dir, f"{asset_name}.tif")
            da.rio.to_raster(tif_path)

        meta = {
            "id": self.id,
            "datetime": self.datetime.isoformat(),
            "platform": self.platform,
            "orbit_direction": self.orbit_direction,
            "crs": self.crs,
            "pixel_size_m": self.pixel_size_m,
            "assets": list(self.data.keys()),
        }
        with open(os.path.join(pass_dir, "metadata.json"), "w") as f:
            _json.dump(meta, f, indent=2)

        self.pass_dir = pass_dir
        return pass_dir

    def load(self) -> "LoadedItem":
        """Load data from disk into memory.

        No-op if data is already loaded or *pass_dir* is not set.
        Raises ``RuntimeError`` if the on-disk files are corrupt.
        """
        if not self.pass_dir or self.data:
            return self

        import rioxarray  # noqa: F401

        meta_path = os.path.join(self.pass_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
            assets = meta.get("assets", [])
        else:
            assets = [
                os.path.splitext(f)[0]
                for f in sorted(os.listdir(self.pass_dir))
                if f.endswith(".tif")
            ]

        for asset_name in assets:
            tif_path = os.path.join(self.pass_dir, f"{asset_name}.tif")
            if os.path.exists(tif_path):
                da = rioxarray.open_rasterio(tif_path, masked=True)
                if "band" in da.dims and da.sizes["band"] == 1:
                    da = da.squeeze("band", drop=True)
                da = da.load()
                # Basic corruption check: must have non-zero shape and
                # not be entirely NaN/zero.
                if da.size == 0:
                    raise RuntimeError(
                        f"Corrupt GeoTIFF (empty): {tif_path}"
                    )
                self.data[asset_name] = da
        return self

    def is_valid_on_disk(self) -> bool:
        """Check whether the on-disk files are present and readable.

        Returns *False* if metadata.json is missing, any expected
        GeoTIFF is missing or cannot be opened by rasterio.
        """
        if not self.pass_dir or not os.path.isdir(self.pass_dir):
            return False
        meta_path = os.path.join(self.pass_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return False
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
        except (_json.JSONDecodeError, OSError):
            return False
        for asset_name in meta.get("assets", []):
            tif_path = os.path.join(self.pass_dir, f"{asset_name}.tif")
            if not os.path.isfile(tif_path) or os.path.getsize(tif_path) == 0:
                return False
            # Quick header check via rasterio
            try:
                import rasterio
                with rasterio.open(tif_path) as src:
                    if src.width == 0 or src.height == 0:
                        return False
            except Exception:
                return False
        return True

    def unload(self) -> None:
        """Free data arrays from memory (keeps *pass_dir* for reloading)."""
        for da in self.data.values():
            try:
                da.close()
            except Exception:
                pass
        self.data.clear()


def parse_opera_rtc_id(item_id: str) -> Dict[str, Any]:
    """Extract metadata from an OPERA RTC-S1 item ID.

    Example ID::

        OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0

    The first timestamp is the acquisition time; the second is the
    processing time.  When duplicate bursts exist (reprocessing),
    the one with the **latest** processing time should be kept.
    """
    result: Dict[str, Any] = {}
    parts = item_id.split("_")

    for part in parts:
        if part in _SENSOR_NAMES:
            result["sensor"] = part
            result["platform"] = _SENSOR_NAMES[part]

    for part in parts:
        m = re.match(r"T(\d+)-(\d+)-(IW\d)", part)
        if m:
            result["track"] = int(m.group(1))
            result["burst"] = int(m.group(2))
            result["swath"] = m.group(3)
            break

    # Extract both timestamps: acquisition (1st) and processing (2nd)
    timestamps = []
    for part in parts:
        m = re.match(r"(\d{8}T\d{6})Z", part)
        if m:
            timestamps.append(
                datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
            )
    if timestamps:
        result["acq_time"] = timestamps[0]
    if len(timestamps) >= 2:
        result["proc_time"] = timestamps[1]

    return result


def deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate OPERA RTC burst items, keeping the latest processed.

    OPERA products are sometimes reprocessed, producing multiple items
    for the same burst (same track, burst number, swath, and acquisition
    time) but with different processing timestamps.  This function keeps
    only the item with the **latest** processing timestamp for each
    unique burst.

    Non-OPERA items (or items whose IDs cannot be parsed) are passed
    through unchanged.

    Parameters
    ----------
    items : list[dict]
        Raw STAC item dictionaries.

    Returns
    -------
    list[dict]
        Deduplicated items in the same relative order.
    """
    # Build a key → (best_proc_time, item) mapping
    best: Dict[str, tuple] = {}        # burst_key → (proc_time, item)
    non_opera: List[Dict[str, Any]] = []

    for item in items:
        item_id = item.get("id", "")
        parsed = parse_opera_rtc_id(item_id)

        track = parsed.get("track")
        burst = parsed.get("burst")
        swath = parsed.get("swath")
        acq = parsed.get("acq_time")

        if track is None or burst is None or swath is None or acq is None:
            non_opera.append(item)
            continue

        burst_key = f"T{track:03d}-{burst:06d}-{swath}_{acq:%Y%m%dT%H%M%S}"
        proc_time = parsed.get("proc_time", datetime.min)

        if burst_key not in best or proc_time > best[burst_key][0]:
            best[burst_key] = (proc_time, item)

    kept = [entry[1] for entry in best.values()]
    n_removed = len(items) - len(kept) - len(non_opera)
    if n_removed > 0:
        logger.info(
            "Deduplicated %d items → %d (removed %d reprocessed duplicates)",
            len(items), len(kept) + len(non_opera), n_removed,
        )

    return non_opera + kept


def extract_item_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardised metadata from a STAC item dictionary."""
    props = item.get("properties", {})
    meta: Dict[str, Any] = {"id": item.get("id", "")}

    dt_str = props.get("datetime") or props.get("start_datetime", "")
    if dt_str:
        meta["datetime"] = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    else:
        meta["datetime"] = None

    orbit = props.get("sat:orbit_state") or props.get("orbit_direction")
    meta["orbit_direction"] = orbit.lower() if orbit else None

    platform = props.get("platform") or props.get("constellation", "")
    parsed = parse_opera_rtc_id(meta["id"])
    if not platform and "platform" in parsed:
        platform = parsed["platform"]
    meta["platform"] = platform or "Unknown"
    meta["parsed"] = parsed

    return meta


def _configure_gdal() -> None:
    """Set GDAL environment variables for efficient COG access."""
    settings = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "NO",
        "GDAL_HTTP_VERSION": "1.1",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "67108864",  # 64 MB
        "GDAL_CACHEMAX": "256",  # MB
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)


def setup_terrascope_auth(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Authenticate with Terrascope using HTTP Basic Auth.

    Credentials are read from the ``TERRASCOPE_USERNAME`` and
    ``TERRASCOPE_PASSWORD`` environment variables if not provided.

    Parameters
    ----------
    username, password : str, optional
        Terrascope credentials.
    """
    from rs_tools.archives.auth import login

    login(username, password, quiet=True)


def load_stac_asset(
    href: str,
    bbox: Optional[BoundingBox] = None,
    s3_url: Optional[str] = None,
    local_url: Optional[str] = None,
    chunks: Optional[dict] = "auto",
):
    """Load a single COG asset as a geo-located xarray DataArray.

    When ``chunks`` is set (default ``"auto"``), the data is loaded
    lazily via Dask — only metadata is read initially and pixel data
    is fetched on demand when ``.values`` or ``.compute()`` is called.
    This dramatically reduces peak memory for long time-series.

    Parameters
    ----------
    href : str
        URL or file path of the COG asset.
    bbox : BoundingBox, optional
        If provided, clip the data to this bounding box.
    s3_url : str, optional
        Corresponding ``s3://`` URI.  When running on AWS this is
        used for faster ``/vsis3/`` access instead of HTTPS.
    local_url : str, optional
        Local ``file://`` URL from the STAC ``alternate.local.href``
        field.  When running on a VITO server with ``/data/MTDA``
        mounts, the local path is used instead of HTTPS.
    chunks : dict or str or None
        Chunk specification for Dask-backed lazy loading.
        ``"auto"`` (default) lets xarray/dask choose chunk sizes.
        Set to ``None`` to load data eagerly into memory.

    Returns
    -------
    xr.DataArray
        Geo-located data array with CRS and spatial coordinates.
    """
    import rioxarray  # noqa: F401

    _configure_gdal()

    # Priority: local path (VITO) > S3 (AWS) > HTTPS
    _resolved_local = False
    if local_url:
        from rs_tools.archives.local import is_on_vito, resolve_local_href
        if is_on_vito():
            local_path = resolve_local_href(local_url)
            if local_path:
                href = local_path
                _resolved_local = True

    # Resolve the best GDAL path: /vsis3/ on AWS, /vsicurl/ otherwise
    if not _resolved_local and (
        "datapool.asf.alaska.edu" in href or (s3_url and "s3://" in s3_url)
    ):
        from rs_tools.archives.nasa import resolve_nasa_href
        href = resolve_nasa_href(href, s3_url=s3_url)

    # Build open_rasterio kwargs — use Dask if chunks is specified
    open_kwargs = {"masked": True}
    if chunks is not None:
        try:
            import dask  # noqa: F401
            open_kwargs["chunks"] = chunks
        except ImportError:
            logger.debug(
                "Dask not installed — loading eagerly. "
                "Install dask for lazy loading: pip install dask"
            )

    # Retry with exponential backoff on HTTP 429 (rate limit) or transient errors
    _max_retries = 5
    _delay = 2.0
    for _attempt in range(_max_retries):
        try:
            da = rioxarray.open_rasterio(href, **open_kwargs)
            break
        except Exception as exc:
            _msg = str(exc)
            _is_rate_limit = "429" in _msg or "rate" in _msg.lower()
            _is_transient = _is_rate_limit or "timeout" in _msg.lower() or "503" in _msg
            if _is_transient and _attempt < _max_retries - 1:
                wait = _delay * (2 ** _attempt)
                logger.debug("HTTP %s on attempt %d/%d, retrying in %.0fs …",
                             "429" if _is_rate_limit else "error",
                             _attempt + 1, _max_retries, wait)
                time.sleep(wait)
            else:
                raise

    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)

    if bbox is not None:
        da = da.rio.clip_box(
            minx=bbox.west,
            miny=bbox.south,
            maxx=bbox.east,
            maxy=bbox.north,
            crs="EPSG:4326",
        )

    # When eager loading (no Dask), force data into memory now so that
    # corrupt tiles are caught here inside the caller's try/except rather
    # than later during mosaic/composite when they are harder to recover from.
    if chunks is None:
        da = da.load()

    return da


def _load_single_item(
    item: Dict[str, Any],
    assets: List[str],
    bbox: Optional[BoundingBox],
    chunks,
    item_idx: int,
    total: int,
) -> Optional[LoadedItem]:
    """Load one STAC item's assets into a LoadedItem.

    Returns *None* if no assets could be loaded.
    """
    meta = extract_item_metadata(item)
    item_assets = item.get("assets", {})
    available = [a for a in assets if a in item_assets]
    if not available:
        logger.warning("Item %s has none of %s, skipping.", meta["id"], assets)
        return None

    data = {}
    crs = None
    pixel_size = None

    for asset_name in available:
        asset_entry = item_assets[asset_name]
        href = asset_entry.get("href", "")
        alternate = asset_entry.get("alternate")

        s3_url = None
        local_url = None
        if isinstance(alternate, str):
            s3_url = alternate
        elif isinstance(alternate, dict):
            from rs_tools.archives.local import extract_local_url
            local_url = extract_local_url(alternate)

        if not href:
            continue
        try:
            da = load_stac_asset(
                href,
                bbox=bbox,
                s3_url=s3_url,
                local_url=local_url,
                chunks=chunks,
            )
            data[asset_name] = da
            if crs is None and da.rio.crs is not None:
                crs = str(da.rio.crs)
                transform = da.rio.transform()
                pixel_size = abs(transform.a)
        except Exception:
            logger.warning(
                "Failed to load asset %s for item %d/%d (%s)",
                asset_name, item_idx + 1, total, meta["id"],
                exc_info=True,
            )

    if not data:
        return None

    print(
        f"  [{item_idx + 1}/{total}] {meta['platform']}"
        f" | {(meta['orbit_direction'] or '?')[:3].upper()}"
        f" | {meta['datetime']:%Y-%m-%d %H:%M UTC}"
    )
    return LoadedItem(
        id=meta["id"],
        datetime=meta["datetime"],
        platform=meta["platform"],
        orbit_direction=meta["orbit_direction"],
        data=data,
        crs=crs,
        pixel_size_m=pixel_size,
    )


def load_items(
    items: List[Dict[str, Any]],
    assets: List[str],
    bbox: Optional[BoundingBox] = None,
    mosaic: bool = False,
    chunks: Optional[dict] = "auto",
    output_dir: Optional[str] = None,
) -> List[LoadedItem]:
    """Load COG assets from STAC items into geo-located DataArrays.

    Items are always loaded **one satellite pass at a time** to keep
    peak memory proportional to a single pass (~2–3 bursts) rather
    than the entire time-series.  Within each pass, burst-level
    assets are loaded eagerly, optionally merged (when *mosaic* is
    True), clipped to *bbox*, and then freed before the next pass.

    When *output_dir* is provided, each mosaicked pass is written to
    disk as GeoTIFF files and then freed from memory.  The returned
    ``LoadedItem`` objects have their pixel data cleared (``data={}``);
    call ``item.load()`` to page data back in from disk when needed.

    For items that cannot be grouped by pass (non-OPERA data), each
    item is treated as its own "pass" — the streaming logic still
    applies.

    Parameters
    ----------
    items : list[dict]
        STAC item dictionaries (as returned by search).
    assets : list[str]
        Asset names to load (e.g. ``["VV", "VH"]``).
    bbox : BoundingBox, optional
        Clip data to this bounding box.
    mosaic : bool
        If *True*, mosaic burst-level items from the same satellite
        pass into single images.
    chunks : dict or str or None
        Chunk specification for Dask-backed lazy loading.
        ``"auto"`` (default) lets xarray/dask choose chunk sizes.
        Set to ``None`` to load data eagerly into memory.
    output_dir : str, optional
        When set, write each mosaicked pass to this directory as
        GeoTIFF files (``<output_dir>/passes/<pass_key>/VV.tif``
        etc.) and free pixel data from memory.  Call
        ``item.load()`` to read data back when needed.

    Returns
    -------
    list[LoadedItem]
        Successfully loaded items with data and metadata.
    """
    from rs_tools.datasets.mosaic import mosaic_items

    # Remove reprocessed duplicates before loading pixel data
    items = deduplicate_items(items)

    # Group STAC items by satellite pass before loading any pixel data.
    # Non-OPERA items land under "unknown" and are each treated individually.
    pass_groups = _group_items_by_pass(items)
    total_items = len(items)
    item_counter = 0

    # Build an index of existing valid passes on disk so we can skip them.
    _existing: Dict[str, LoadedItem] = {}
    if output_dir:
        passes_dir = os.path.join(output_dir, "passes")
        if os.path.isdir(passes_dir):
            for entry in os.listdir(passes_dir):
                meta_path = os.path.join(passes_dir, entry, "metadata.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path) as _f:
                        _m = _json.load(_f)
                    _item = LoadedItem(
                        id=_m["id"],
                        datetime=datetime.fromisoformat(_m["datetime"]),
                        platform=_m["platform"],
                        orbit_direction=_m.get("orbit_direction"),
                        data={},
                        crs=_m.get("crs"),
                        pixel_size_m=_m.get("pixel_size_m"),
                        pass_dir=os.path.join(passes_dir, entry),
                    )
                    if _item.is_valid_on_disk():
                        _existing[entry] = _item
                except Exception:
                    pass
        if _existing:
            print(f"  Found {len(_existing)} existing valid passes on disk")

    result: List[LoadedItem] = []

    sorted_keys = sorted(pass_groups.keys())
    for pass_idx, pass_key in enumerate(sorted_keys):
        pass_items = pass_groups[pass_key]

        # Predict the pass_key that save() would produce, so we can
        # check whether this pass already exists on disk.
        if output_dir:
            _probe_meta = extract_item_metadata(pass_items[0])
            _probe_dt = _probe_meta.get("datetime")
            _probe_orb = _probe_meta.get("orbit_direction")
            _probe_parsed = _probe_meta.get("parsed", {})
            if _probe_dt is not None:
                _dt_str = _probe_dt.strftime("%Y%m%d_%H%M")
                _orb_str = (_probe_orb or "UNK")[:3].upper()
                _track = _probe_parsed.get("track")
                _key_parts = [_dt_str, _orb_str]
                if _track is not None:
                    _key_parts.append(f"T{_track:03d}")
                _predicted_key = "_".join(_key_parts)
                if _predicted_key in _existing:
                    print(f"  ⏭ {_predicted_key}: already on disk, skipping")
                    result.append(_existing[_predicted_key])
                    item_counter += len(pass_items)
                    continue

        # Load each burst in this pass eagerly — no Dask chunks — so
        # that only one pass worth of data is in memory at a time.
        burst_loaded: List[LoadedItem] = []
        for stac_item in pass_items:
            li = _load_single_item(
                stac_item, assets,
                bbox=None if mosaic else bbox,  # defer clip when merging
                chunks=None,                    # eager: read pixels now
                item_idx=item_counter,
                total=total_items,
            )
            if li is not None:
                burst_loaded.append(li)
            item_counter += 1
            time.sleep(0.3)

        if not burst_loaded:
            continue

        if mosaic:
            # Merge multi-burst passes, clip to bbox
            mosaicked = mosaic_items(burst_loaded, bbox=bbox)
            # Free burst-level arrays
            for bli in burst_loaded:
                for da in bli.data.values():
                    try:
                        da.close()
                    except Exception:
                        pass
                bli.data.clear()
            # Save to disk and free memory when output_dir is set
            if output_dir:
                for m_item in mosaicked:
                    pdir = m_item.save(output_dir)
                    m_item.unload()
                    print(f"    → saved to {pdir}")
            result.extend(mosaicked)
        else:
            if output_dir:
                for bli in burst_loaded:
                    pdir = bli.save(output_dir)
                    bli.unload()
                    print(f"    → saved to {pdir}")
            result.extend(burst_loaded)

        # Brief pause between passes to be kind to the server
        if pass_idx < len(sorted_keys) - 1:
            time.sleep(0.5)

    result.sort(key=lambda x: x.datetime or datetime.min)
    return result


def _group_items_by_pass(
    items: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group STAC items by satellite pass (track + date).

    Returns a dict keyed by ``"TRRR_YYYY-MM-DD"`` with lists of items.
    Items whose ID cannot be parsed are placed under key ``"unknown"``.
    """
    from collections import defaultdict
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        parsed = parse_opera_rtc_id(item.get("id", ""))
        track = parsed.get("track")
        acq = parsed.get("acq_time")
        if track is not None and acq is not None:
            key = f"T{track:03d}_{acq:%Y-%m-%d}"
        else:
            key = "unknown"
        groups[key].append(item)
    return dict(groups)


def load_passes_from_disk(output_dir: str) -> List[LoadedItem]:
    """Reload pass metadata from a previous :func:`load_items` run.

    Returns ``LoadedItem`` objects with ``data={}`` (no pixel data in
    memory).  Call ``item.load()`` to page data in from the GeoTIFF
    files on disk when you need it, and ``item.unload()`` to free
    memory afterwards.

    Parameters
    ----------
    output_dir : str
        The same directory that was passed as *output_dir* to
        :func:`load_items`.

    Returns
    -------
    list[LoadedItem]
        Metadata-only items sorted by datetime.
    """
    passes_dir = os.path.join(output_dir, "passes")
    if not os.path.isdir(passes_dir):
        return []

    items: List[LoadedItem] = []
    n_corrupt = 0
    for entry in sorted(os.listdir(passes_dir)):
        pass_dir = os.path.join(passes_dir, entry)
        meta_path = os.path.join(pass_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
        except (_json.JSONDecodeError, OSError):
            n_corrupt += 1
            logger.warning("Corrupt metadata in %s, skipping", pass_dir)
            continue
        item = LoadedItem(
            id=meta["id"],
            datetime=datetime.fromisoformat(meta["datetime"]),
            platform=meta["platform"],
            orbit_direction=meta.get("orbit_direction"),
            data={},
            crs=meta.get("crs"),
            pixel_size_m=meta.get("pixel_size_m"),
            pass_dir=pass_dir,
        )
        if not item.is_valid_on_disk():
            n_corrupt += 1
            logger.warning("Corrupt or incomplete pass in %s, skipping", pass_dir)
            continue
        items.append(item)

    items.sort(key=lambda x: x.datetime)
    msg = f"Found {len(items)} saved passes in {passes_dir}"
    if n_corrupt:
        msg += f" ({n_corrupt} corrupt/incomplete skipped)"
    print(msg)
    return items


def subsample_monthly(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one satellite pass per calendar month (pre-download).

    Groups burst-level STAC items by pass (track + date), then picks
    the pass with the **most bursts** per calendar month.  All bursts
    belonging to the selected pass are kept so mosaicking still works.

    Parameters
    ----------
    items : list[dict]
        STAC item dicts as returned by search.

    Returns
    -------
    list[dict]
        Subset of items covering ~1 pass per month (best-coverage pass).
    """
    from collections import defaultdict as _dd

    passes = _group_items_by_pass(items)

    sorted_keys = sorted(k for k in passes if k != "unknown")

    # Group all candidate passes by calendar month
    month_candidates: dict = _dd(list)
    for key in sorted_keys:
        month_key = key.split("_", 1)[1][:7]  # "2024-06"
        month_candidates[month_key].append(key)

    # For each month pick the pass that has the most bursts (best coverage)
    keep_keys: List[str] = [
        max(keys, key=lambda k: len(passes[k]))
        for keys in month_candidates.values()
    ]

    result: List[Dict[str, Any]] = []
    for key in keep_keys:
        result.extend(passes[key])
    # Also keep unknown items
    if "unknown" in passes:
        result.extend(passes["unknown"])
    return result


def _search_monthly(
    archive: str,
    config: "SearchConfig",
    collections: List[str],
    bbox: "BoundingBox",
    interval_months: int = 1,
) -> List[Dict[str, Any]]:
    """Search month-by-month, keeping one pass per *interval_months*.

    Avoids hitting the item limit on long time ranges with large AOIs
    by searching each month individually and selecting the first
    available pass.

    Parameters
    ----------
    interval_months : int
        Keep one pass every *interval_months* calendar months.
        Default is 1 (every month).  Set to 6 for bi-annual sampling.
    """
    from dateutil.relativedelta import relativedelta

    start = (
        datetime.strptime(str(config.start_date), "%Y-%m-%d").date()
        if not hasattr(config.start_date, "year")
        else config.start_date
    )
    end = (
        datetime.strptime(str(config.end_date), "%Y-%m-%d").date()
        if not hasattr(config.end_date, "year")
        else config.end_date
    )

    all_items: List[Dict[str, Any]] = []
    cursor = start.replace(day=1)

    while cursor <= end:
        month_end = (cursor + relativedelta(months=interval_months)) - relativedelta(days=1)
        if month_end > end:
            month_end = end

        month_config = SearchConfig(
            start_date=str(cursor),
            end_date=str(month_end),
            bbox=bbox,
            collections=collections,
            limit=config.limit,
        )
        month_items = search_archive(archive, month_config)

        if month_items:
            selected = subsample_monthly(month_items)
            all_items.extend(selected)
            _dates = sorted({
                parse_opera_rtc_id(i.get("id", ""))["acq_time"].strftime("%Y-%m-%d")
                for i in selected
                if parse_opera_rtc_id(i.get("id", "")).get("acq_time")
            })
            print(f"  {cursor:%Y-%m}: {len(month_items)} found → "
                  f"{len(selected)} bursts on {len(_dates)} date(s): "
                  f"{', '.join(_dates)}")
        else:
            print(f"  {cursor:%Y-%m}: no data")

        cursor += relativedelta(months=interval_months)

    return all_items


def _search_with_priority(
    archive_priority: List[str],
    ds_info,
    start_date,
    end_date,
    bbox,
    limit: int,
    monthly: bool = False,
    interval_months: int = 1,
) -> tuple:
    """Try archives in *archive_priority* order, return first with results.

    Returns ``(archive_name, items)``.
    """
    items: List[Dict[str, Any]] = []
    archive = archive_priority[0]

    for candidate in archive_priority:
        collections = ds_info.archive_collections.get(candidate, [])
        if not collections:
            continue
        config = SearchConfig(
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            collections=collections,
            limit=limit,
        )
        try:
            if monthly:
                items = _search_monthly(
                    candidate, config, collections, bbox, interval_months,
                )
            else:
                items = search_archive(candidate, config)
        except Exception:
            logger.warning(
                "Search failed for archive %s, trying next", candidate,
                exc_info=True,
            )
            items = []

        if items:
            archive = candidate
            logger.info(
                "Archive %s returned %d items", candidate, len(items),
            )
            break
        logger.info("Archive %s returned 0 items, trying next", candidate)

    return archive, items


def load_dataset(
    short_name: str,
    bbox: BoundingBox,
    start_date: Union[str, "date"],
    end_date: Union[str, "date"],
    archive: Optional[str] = None,
    assets: Optional[List[str]] = None,
    limit: int = 20,
    mosaic: bool = True,
    monthly: bool = False,
    interval_months: int = 1,
    chunks: Optional[dict] = "auto",
    output_dir: Optional[str] = None,
) -> List[LoadedItem]:
    """Search and load a known dataset as geo-located data.

    High-level function that combines catalog lookup, STAC search,
    and COG loading into a single call.

    Parameters
    ----------
    short_name : str
        Dataset identifier from the catalog (e.g. ``"OPERA_RTC_S1"``).
    bbox : BoundingBox
        Spatial bounding box.
    start_date, end_date : str or date
        Temporal search window.
    archive : str, optional
        Which archive(s) to search:

        * ``"terrascope"`` — Terrascope STAC only.
        * ``"nasa"`` — NASA ASF only.
        * ``"cdse"`` — Copernicus Data Space only.
        * ``None`` (default) — try archives in catalog priority order
          (Terrascope first, then NASA/ASF) and use the first one that
          returns results.
    assets : list[str], optional
        Asset names to load. Uses dataset defaults if *None*.
    limit : int
        Maximum items to load.
    mosaic : bool
        If *True* (default), mosaic burst-level items from the same
        satellite pass into single images.
    monthly : bool
        If *True*, subsample to one pass per calendar month (or per
        *interval_months* months) before downloading.
    interval_months : int
        When *monthly=True*, keep one pass every *interval_months*
        calendar months.  Default 1 (monthly).  Use 6 for bi-annual.
    chunks : dict or str or None
        Chunk specification for Dask-backed lazy loading.
        ``"auto"`` (default) lets xarray/dask choose chunk sizes.
        Set to ``None`` to load data eagerly into memory.
    output_dir : str, optional
        When set, write each mosaicked pass to this directory and
        free pixel data from memory.  See :func:`load_items`.

    Returns
    -------
    list[LoadedItem]
        Loaded items sorted by datetime.
    """
    ds_info = get_dataset(short_name)

    # -----------------------------------------------------------------
    # Archive resolution: explicit name or prioritised fallback
    # -----------------------------------------------------------------
    if archive is None:
        archive_priority = list(ds_info.archive_collections.keys())
        archive, items = _search_with_priority(
            archive_priority, ds_info, start_date, end_date, bbox, limit,
            monthly, interval_months,
        )
        if assets is None:
            assets = _DEFAULT_ASSETS.get(short_name)
        print(f"Found {len(items)} items for {short_name} in {archive}")
        if not items:
            return []
        return _configure_and_load(
            archive, items, assets, bbox, mosaic, chunks,
            ds_info, short_name, output_dir=output_dir,
        )

    collections = ds_info.archive_collections.get(archive, [])
    if not collections:
        raise ValueError(
            f"Dataset {short_name!r} not available in archive {archive!r}"
        )

    if assets is None:
        assets = _DEFAULT_ASSETS.get(short_name)

    config = SearchConfig(
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        collections=collections,
        limit=limit,
    )

    if monthly:
        items = _search_monthly(archive, config, collections, bbox,
                                interval_months=interval_months)
    else:
        items = search_archive(archive, config)

    print(f"Found {len(items)} items for {short_name} in {archive}")

    if not items:
        return []

    return _configure_and_load(archive, items, assets, bbox, mosaic, chunks,
                               ds_info, short_name, output_dir=output_dir)


def _configure_and_load(
    archive: str,
    items: List[Dict[str, Any]],
    assets: Optional[List[str]],
    bbox: "BoundingBox",
    mosaic: bool,
    chunks,
    ds_info=None,
    short_name: str = "",
    output_dir: Optional[str] = None,
) -> List["LoadedItem"]:
    """Configure GDAL auth for *archive*, resolve assets, and load items."""

    # Configure GDAL auth for the target archive
    if archive == "nasa":
        from rs_tools.archives.nasa import configure_gdal_nasa
        configure_gdal_nasa()
    elif archive == "cdse":
        from rs_tools.archives.cdse import configure_gdal_cdse
        configure_gdal_cdse()
    elif archive == "terrascope":
        # On VITO servers data is read from local mounts — HTTPS auth is
        # only needed as fallback when local paths are unavailable.
        from rs_tools.archives.local import is_on_vito
        if is_on_vito():
            logger.info("On VITO server — will use local paths if available")
            try:
                setup_terrascope_auth()
            except RuntimeError:
                logger.info(
                    "Terrascope HTTPS credentials not configured — "
                    "local file access will be used"
                )
        else:
            setup_terrascope_auth()

    # Determine assets from first item if not specified
    if assets is None:
        assets = _DEFAULT_ASSETS.get(short_name)
    if assets is None:
        first = items[0].get("assets", {})
        assets = [
            k
            for k, v in first.items()
            if "image" in v.get("type", "") or k in ("VV", "VH", "data")
        ]

    return load_items(items, assets=assets, bbox=bbox, mosaic=mosaic, chunks=chunks,
                       output_dir=output_dir)
