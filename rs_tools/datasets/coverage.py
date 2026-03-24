"""Coverage analysis and metadata reporting for raw search results.

Provides tools to inspect STAC search results **before** loading pixel
data.  The typical workflow is:

1. Search one or more archives with :func:`~rs_tools.search.search_archive`.
2. Call :func:`summarize_search_results` to build a per-pass coverage table.
3. Inspect the table with :func:`print_coverage_report`.
4. Call :func:`filter_by_coverage` to keep only the passes you want.
5. Convert back to raw items with :func:`records_to_items` and pass to
   :func:`~rs_tools.datasets.loader.load_items` for lazy or eager loading.

Example::

    from rs_tools.config import BoundingBox, SearchConfig
    from rs_tools.search import search_archive
    from rs_tools.datasets.coverage import (
        summarize_search_results,
        print_coverage_report,
        filter_by_coverage,
        records_to_items,
    )
    from rs_tools.datasets.loader import load_items

    bbox = BoundingBox(west=-118.5, south=34.0, east=-117.5, north=35.0)
    config = SearchConfig(
        start_date="2024-01-01",
        end_date="2024-06-30",
        bbox=bbox,
        collections=["OPERA_L2_RTC-S1_V1_1"],
        limit=500,
    )
    items = search_archive("nasa", config)

    records = summarize_search_results(items, bbox)
    print_coverage_report(records)

    selected = filter_by_coverage(records, min_coverage_pct=80.0, orbit_direction="ascending")
    loaded = load_items(records_to_items(selected), assets=["VV", "VH"], bbox=bbox, mosaic=True)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from rs_tools.config import BoundingBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PassRecord — one record per satellite pass
# ---------------------------------------------------------------------------

@dataclass
class PassRecord:
    """Summary record for a single satellite pass or product acquisition.

    A *pass* groups all STAC granules (bursts / tiles) from the same
    satellite overflight: same track number + same acquisition date for
    OPERA-style burst products; same platform + orbit direction + date
    for generic STAC items.

    Attributes
    ----------
    date : str
        ISO date string ``"YYYY-MM-DD"`` of the acquisition.
    utc_time : str
        UTC time string ``"HH:MM:SS"`` of the acquisition.
    platform : str
        Satellite platform name, e.g. ``"Sentinel-1A"``.
    orbit_direction : str or None
        Flight direction: ``"ascending"`` (south→north) or
        ``"descending"`` (north→south).  *None* if unknown.
    track : int or None
        Relative orbit (track) number — the repeating ground-track
        identifier (e.g. 59 for T059).  *None* for non-SAR data.
    n_granules : int
        Number of STAC items (bursts / tiles) belonging to this pass.
    coverage_pct : float
        Percentage of the query bbox area covered by the union of all
        item geometries (0–100).  ``nan`` if geometry was unavailable
        or shapely is not installed.
    https_urls : list[str]
        HTTPS asset download / streaming links from all granules.
    s3_urls : list[str]
        ``s3://`` asset links from all granules (may be empty).
    stac_item_urls : list[str]
        STAC ``self`` links for each individual granule (may be empty if
        the archive does not embed self links in search results).
    items : list[dict]
        Original STAC item dictionaries — pass these to
        :func:`~rs_tools.datasets.loader.load_items`.
    """

    date: str
    utc_time: str
    platform: str
    orbit_direction: Optional[str]
    track: Optional[int]
    n_granules: int
    coverage_pct: float
    https_urls: List[str] = field(default_factory=list)
    s3_urls: List[str] = field(default_factory=list)
    stac_item_urls: List[str] = field(default_factory=list)
    items: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def label(self) -> str:
        """Human-readable one-line summary of this pass.

        Example: ``"Sentinel-1A | ASC | T059 | 2024-06-30 17:41:51 UTC | cov=92.3%"``
        """
        parts = [self.platform]
        if self.orbit_direction:
            parts.append(self.orbit_direction[:3].upper())
        if self.track is not None:
            parts.append(f"T{self.track:03d}")
        parts.append(f"{self.date} {self.utc_time} UTC")
        if not math.isnan(self.coverage_pct):
            parts.append(f"cov={self.coverage_pct:.1f}%")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pass_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract platform, track, acquisition time and orbit from a STAC item.

    Tries the OPERA RTC item-ID parser first; falls back to generic
    STAC property extraction for other datasets.

    Returns
    -------
    dict with keys ``platform``, ``track``, ``acq_time``, ``orbit_direction``.
    """
    from rs_tools.datasets.loader import parse_opera_rtc_id

    item_id = item.get("id", "")
    props = item.get("properties", {})

    # -- OPERA RTC / CSLC items: parse rich metadata from the item ID ------
    if "OPERA_L2_RTC" in item_id or "OPERA_L2_CSLC" in item_id:
        parsed = parse_opera_rtc_id(item_id)
        track: Optional[int] = parsed.get("track")
        acq_time: Optional[datetime] = parsed.get("acq_time")
        platform: str = (
            props.get("platform")
            or parsed.get("platform", "Unknown")
        )
    else:
        # Generic STAC properties
        track = None
        acq_time = None
        platform = (
            props.get("platform")
            or props.get("constellation")
            or "Unknown"
        )

    # -- Generic datetime extraction (covers the generic case & CDSE items) -
    if acq_time is None:
        dt_str = props.get("datetime") or props.get("start_datetime", "")
        if dt_str:
            try:
                acq_time = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except ValueError:
                logger.debug("Could not parse datetime %r for item %s", dt_str, item_id)

    orbit = (
        props.get("sat:orbit_state")
        or props.get("orbit_direction")
        or ""
    ).lower() or None

    return {
        "platform": platform,
        "track": track,
        "acq_time": acq_time,
        "orbit_direction": orbit,
    }


def _pass_group_key(info: Dict[str, Any]) -> str:
    """Return a string that uniquely identifies the satellite pass.

    For OPERA-style data, uses ``platform|orbit|TRRR|YYYY-MM-DD``.
    For generic data, uses ``platform|orbit|YYYY-MM-DD``.
    """
    platform = info.get("platform", "Unknown")
    orbit = info.get("orbit_direction") or "unknown_orbit"
    acq_time: Optional[datetime] = info.get("acq_time")
    date_str = acq_time.strftime("%Y-%m-%d") if acq_time is not None else "unknown_date"
    track: Optional[int] = info.get("track")

    if track is not None:
        return f"{platform}|{orbit}|T{track:03d}|{date_str}"
    return f"{platform}|{orbit}|{date_str}"


def compute_bbox_coverage(
    geometry: Dict[str, Any],
    bbox: BoundingBox,
) -> float:
    """Compute what percentage of *bbox* is covered by *geometry*.

    Uses the planar (degree²) area ratio, which is a good approximation
    for bboxes smaller than ~10°.  Requires ``shapely``; returns ``nan``
    if shapely is unavailable or the geometry is invalid.

    Parameters
    ----------
    geometry : dict
        GeoJSON geometry dictionary (``{"type": ..., "coordinates": ...}``).
    bbox : BoundingBox
        Query bounding box.

    Returns
    -------
    float
        Coverage percentage in the range 0–100, or ``nan``.
    """
    try:
        from shapely.geometry import box as shapely_box, shape  # noqa: PLC0415
    except ImportError:
        logger.debug("shapely not available — coverage will be reported as nan")
        return float("nan")

    try:
        item_shape = shape(geometry)
        bbox_shape = shapely_box(bbox.west, bbox.south, bbox.east, bbox.north)
        if bbox_shape.area == 0:
            return 0.0
        intersection = item_shape.intersection(bbox_shape)
        return min(100.0, (intersection.area / bbox_shape.area) * 100.0)
    except Exception as exc:
        logger.debug("Coverage computation failed: %s", exc)
        return float("nan")


def _union_coverage(
    group_items: List[Dict[str, Any]],
    bbox: BoundingBox,
) -> float:
    """Return the combined coverage of a group of items over *bbox*.

    Unions all item geometries before intersecting with the bbox so
    overlapping bursts are not double-counted.
    """
    try:
        from shapely.geometry import box as shapely_box, shape  # noqa: PLC0415
        from shapely.ops import unary_union  # noqa: PLC0415
    except ImportError:
        return float("nan")

    geometries = []
    for item in group_items:
        geom = item.get("geometry")
        if geom:
            try:
                geometries.append(shape(geom))
            except Exception as exc:
                logger.debug("Invalid geometry for item %s: %s", item.get("id", "?"), exc)

    if not geometries:
        return float("nan")

    try:
        bbox_shape = shapely_box(bbox.west, bbox.south, bbox.east, bbox.north)
        union_geom = unary_union(geometries)
        if bbox_shape.area == 0:
            return 0.0
        intersection = union_geom.intersection(bbox_shape)
        return min(100.0, (intersection.area / bbox_shape.area) * 100.0)
    except Exception as exc:
        logger.debug("Union coverage computation failed: %s", exc)
        return float("nan")


def _collect_urls(
    group_items: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (https_urls, s3_urls, stac_item_self_links) for a group."""
    https_urls: List[str] = []
    s3_urls: List[str] = []
    stac_item_urls: List[str] = []

    for item in group_items:
        # STAC self link
        for link in item.get("links", []):
            if link.get("rel") == "self":
                href = link.get("href", "")
                if href:
                    stac_item_urls.append(href)
                break

        # Asset URLs
        for asset in item.get("assets", {}).values():
            href: str = asset.get("href", "")
            if isinstance(href, str) and href.startswith("https://"):
                https_urls.append(href)
            alt = asset.get("alternate", "")
            # alternate can be a string (NASA) or a dict (Terrascope STAC)
            if isinstance(alt, str) and alt.startswith("s3://"):
                s3_urls.append(alt)
            elif isinstance(alt, dict):
                for v in alt.values():
                    if isinstance(v, dict):
                        s3_href = v.get("href", "")
                    else:
                        s3_href = v if isinstance(v, str) else ""
                    if isinstance(s3_href, str) and s3_href.startswith("s3://"):
                        s3_urls.append(s3_href)

    return https_urls, s3_urls, stac_item_urls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_search_results(
    items: List[Dict[str, Any]],
    bbox: BoundingBox,
) -> List[PassRecord]:
    """Build a per-pass coverage summary from raw STAC search results.

    Groups STAC items by satellite pass — defined as the same track
    number + acquisition date for OPERA burst products, or the same
    platform + orbit direction + date for generic STAC items.  Computes
    the combined bbox coverage for each pass by taking the union of all
    item geometries.

    Parameters
    ----------
    items : list[dict]
        Raw STAC item dictionaries as returned by
        :func:`~rs_tools.search.search_archive`.
    bbox : BoundingBox
        The spatial bounding box used for the search.

    Returns
    -------
    list[PassRecord]
        One record per pass, sorted by acquisition date and time.
        Pass these to :func:`print_coverage_report` for display or
        to :func:`filter_by_coverage` for selection.
    """
    if not items:
        return []

    # Remove reprocessed duplicates before grouping
    from rs_tools.datasets.loader import deduplicate_items
    items = deduplicate_items(items)

    # Group items by pass key
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    group_rep_info: Dict[str, Dict[str, Any]] = {}  # one info dict per group

    for item in items:
        info = _extract_pass_info(item)
        key = _pass_group_key(info)
        groups[key].append(item)
        if key not in group_rep_info:
            group_rep_info[key] = info

    records: List[PassRecord] = []

    for key in sorted(groups.keys()):
        group_items = groups[key]
        info = group_rep_info[key]

        coverage = _union_coverage(group_items, bbox)
        https_urls, s3_urls, stac_item_urls = _collect_urls(group_items)

        acq_time: Optional[datetime] = info.get("acq_time")
        date_str = acq_time.strftime("%Y-%m-%d") if acq_time is not None else "unknown"
        time_str = acq_time.strftime("%H:%M:%S") if acq_time is not None else "unknown"

        records.append(PassRecord(
            date=date_str,
            utc_time=time_str,
            platform=info.get("platform", "Unknown"),
            orbit_direction=info.get("orbit_direction"),
            track=info.get("track"),
            n_granules=len(group_items),
            coverage_pct=coverage,
            https_urls=https_urls,
            s3_urls=s3_urls,
            stac_item_urls=stac_item_urls,
            items=group_items,
        ))

    records.sort(key=lambda r: (r.date, r.utc_time))
    return records


def print_coverage_report(
    records: List[PassRecord],
    show_urls: bool = False,
) -> None:
    """Print a human-readable coverage report to stdout.

    Parameters
    ----------
    records : list[PassRecord]
        As returned by :func:`summarize_search_results`.
    show_urls : bool
        If *True*, print the HTTPS URL for the first asset of each pass
        below the main table row (useful for quick spot-checking).
    """
    if not records:
        print("No records to display.")
        return

    # --- Header -----------------------------------------------------------
    col_widths = {
        "idx": 4,
        "date": 12,
        "utc": 10,
        "platform": 14,
        "orbit": 5,
        "track": 6,
        "gran": 8,
        "cov": 9,
    }
    header = (
        f"{'#':>{col_widths['idx']}}"
        f"  {'Date':<{col_widths['date']}}"
        f"  {'UTC Time':<{col_widths['utc']}}"
        f"  {'Platform':<{col_widths['platform']}}"
        f"  {'Orbit':<{col_widths['orbit']}}"
        f"  {'Track':<{col_widths['track']}}"
        f"  {'Granules':>{col_widths['gran']}}"
        f"  {'Coverage':>{col_widths['cov']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    has_nan_coverage = False
    for i, rec in enumerate(records, 1):
        if math.isnan(rec.coverage_pct):
            cov_str = "n/a"
            has_nan_coverage = True
        else:
            cov_str = f"{rec.coverage_pct:.1f}%"

        orbit_str = (rec.orbit_direction or "")[:3].upper() or "?"
        track_str = f"T{rec.track:03d}" if rec.track is not None else "-"
        print(
            f"{i:>{col_widths['idx']}}"
            f"  {rec.date:<{col_widths['date']}}"
            f"  {rec.utc_time:<{col_widths['utc']}}"
            f"  {rec.platform:<{col_widths['platform']}}"
            f"  {orbit_str:<{col_widths['orbit']}}"
            f"  {track_str:<{col_widths['track']}}"
            f"  {rec.n_granules:>{col_widths['gran']}}"
            f"  {cov_str:>{col_widths['cov']}}"
        )
        if show_urls and rec.https_urls:
            print(f"      HTTPS : {rec.https_urls[0]}")
        if show_urls and rec.s3_urls:
            print(f"      S3    : {rec.s3_urls[0]}")
        if show_urls and rec.stac_item_urls:
            print(f"      STAC  : {rec.stac_item_urls[0]}")

    print(sep)

    n_known = sum(1 for r in records if not math.isnan(r.coverage_pct))
    if n_known:
        avg = sum(r.coverage_pct for r in records if not math.isnan(r.coverage_pct)) / n_known
        print(
            f"Total: {len(records)} passes  |  "
            f"mean coverage: {avg:.1f}%  |  "
            f"full coverage (≥100%): {sum(1 for r in records if r.coverage_pct >= 100.0)}"
        )
    else:
        print(f"Total: {len(records)} passes")

    if has_nan_coverage:
        print(
            "Note: 'n/a' coverage means item geometry was missing or "
            "shapely is not installed."
        )


def filter_by_coverage(
    records: List[PassRecord],
    min_coverage_pct: float = 0.0,
    orbit_direction: Optional[str] = None,
    track: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    platform: Optional[str] = None,
) -> List[PassRecord]:
    """Filter :class:`PassRecord` objects by coverage and other criteria.

    All criteria are applied with AND logic — a record must satisfy
    every specified filter to be kept.

    Parameters
    ----------
    records : list[PassRecord]
        As returned by :func:`summarize_search_results`.
    min_coverage_pct : float
        Minimum bbox coverage (0–100).  Records with ``nan`` coverage
        are **kept** when this is 0 (the default); set a positive
        threshold to exclude them.
    orbit_direction : str, optional
        Keep only passes with this orbit direction, e.g.
        ``"ascending"`` or ``"descending"`` (case-insensitive).
    track : int, optional
        Keep only passes from this OPERA relative orbit track number.
    start_date : str, optional
        ISO date string ``"YYYY-MM-DD"``.  Drop passes before this date.
    end_date : str, optional
        ISO date string ``"YYYY-MM-DD"``.  Drop passes after this date.
    platform : str, optional
        Keep only passes whose ``platform`` field contains this string
        (case-insensitive substring match).

    Returns
    -------
    list[PassRecord]
        Filtered records in the same order as the input.
    """
    out: List[PassRecord] = []
    for rec in records:
        # Coverage threshold — records with nan coverage are excluded when
        # a positive threshold is given, kept otherwise.
        if min_coverage_pct > 0:
            if math.isnan(rec.coverage_pct) or rec.coverage_pct < min_coverage_pct:
                continue

        if orbit_direction is not None:
            if (rec.orbit_direction or "").lower() != orbit_direction.lower():
                continue

        if track is not None and rec.track != track:
            continue

        if start_date is not None and rec.date < start_date:
            continue

        if end_date is not None and rec.date > end_date:
            continue

        if platform is not None and platform.lower() not in rec.platform.lower():
            continue

        out.append(rec)

    return out


def records_to_items(records: List[PassRecord]) -> List[Dict[str, Any]]:
    """Flatten :class:`PassRecord` objects back to raw STAC item dicts.

    Use this to feed the output of :func:`filter_by_coverage` into
    :func:`~rs_tools.datasets.loader.load_items` or any other function
    that accepts raw STAC item dictionaries.

    Parameters
    ----------
    records : list[PassRecord]
        Selected pass records.

    Returns
    -------
    list[dict]
        Flat list of raw STAC item dicts in pass order (sorted by date).
    """
    result: List[Dict[str, Any]] = []
    for rec in records:
        result.extend(rec.items)
    return result
