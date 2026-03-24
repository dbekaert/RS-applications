"""Mosaicking module for combining same-pass burst data.

OPERA RTC-S1 products are delivered as individual burst granules.
When a bounding box spans multiple bursts from the same satellite
pass, this module merges them into a single continuous image per
pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from rs_tools.datasets.loader import LoadedItem, parse_opera_rtc_id

logger = logging.getLogger(__name__)


def _group_key(item: LoadedItem) -> Optional[Tuple[int, str]]:
    """Return (track, date_string) grouping key for an OPERA RTC item.

    Items from the same track acquired on the same date belong to the
    same satellite pass and should be mosaicked together.

    Returns *None* for items whose ID cannot be parsed.
    """
    parsed = parse_opera_rtc_id(item.id)
    track = parsed.get("track")
    acq_time = parsed.get("acq_time")
    if track is None or acq_time is None:
        return None
    return (track, acq_time.strftime("%Y-%m-%d"))


def group_by_pass(items: List[LoadedItem]) -> Dict[Tuple[int, str], List[LoadedItem]]:
    """Group loaded items by satellite pass (track + date).

    Parameters
    ----------
    items : list[LoadedItem]
        Loaded burst-level items.

    Returns
    -------
    dict
        Mapping of ``(track, date_str)`` to lists of items belonging
        to that pass.
    """
    groups: Dict[Tuple[int, str], List[LoadedItem]] = defaultdict(list)
    for item in items:
        key = _group_key(item)
        if key is None:
            logger.warning("Cannot determine pass for item %s, skipping", item.id)
            continue
        groups[key].append(item)
    return dict(groups)


def _merge_arrays(arrays):
    """Merge a list of xarray DataArrays into one using rioxarray.

    If a merge fails (e.g. corrupt tile in a COG), bad arrays are
    identified and removed one by one until merge succeeds or only a
    single array remains.

    Lazy (Dask-backed) arrays are eagerly materialised *before*
    merging so that transient HTTP errors are caught early and can
    be retried individually, rather than failing the whole merge.
    """
    import time
    import dask.array
    from rioxarray.merge import merge_arrays

    # Materialize lazy arrays one-by-one with retries
    materialised = []
    for i, arr in enumerate(arrays):
        if hasattr(arr.data, 'dask_graph') or isinstance(arr.data, dask.array.Array):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    arr = arr.load()
                    break
                except Exception as exc:
                    if attempt < max_retries - 1:
                        wait = 2 * (2 ** attempt)
                        logger.warning(
                            "Failed to load array %d/%d (attempt %d/%d): %s — "
                            "retrying in %ds",
                            i + 1, len(arrays), attempt + 1, max_retries,
                            exc, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.warning(
                            "Skipping array %d/%d after %d failed attempts: %s",
                            i + 1, len(arrays), max_retries, exc,
                        )
                        arr = None
                        break
        if arr is not None:
            materialised.append(arr)

    if not materialised:
        raise RuntimeError("All arrays failed to load — cannot merge")

    if len(materialised) == 1:
        return materialised[0]

    attempt = list(materialised)
    while attempt:
        try:
            merged = merge_arrays(attempt)
            # Free source arrays — merged result is independent
            for arr in materialised:
                try:
                    arr.close()
                except Exception:
                    pass
            del materialised, attempt
            return merged
        except Exception as exc:
            if len(attempt) == 1:
                # Last array is itself corrupt — re-raise
                raise
            logger.warning(
                "merge_arrays failed (%s); dropping last of %d arrays and retrying",
                exc, len(attempt),
            )
            attempt = attempt[:-1]
    raise RuntimeError("All arrays were corrupt — cannot merge")


def _clip_item_to_bbox(item: LoadedItem, bbox) -> LoadedItem:
    """Clip all DataArrays in a LoadedItem to a BoundingBox (in-place)."""
    for asset_name, da in item.data.items():
        try:
            item.data[asset_name] = da.rio.clip_box(
                minx=bbox.west,
                miny=bbox.south,
                maxx=bbox.east,
                maxy=bbox.north,
                crs="EPSG:4326",
            )
        except Exception as exc:
            logger.warning("clip_box failed for %s asset %s: %s", item.id, asset_name, exc)
    return item


def _snap_to_reference(
    items: List[LoadedItem],
    ref_item: LoadedItem,
) -> List[LoadedItem]:
    """Reproject all items to match the first item's grid exactly.

    After ``clip_box``, different passes may have slightly different
    pixel grids (±1 pixel offset due to different native UTM origins).
    This function snaps every item to the reference item's exact grid
    using ``reproject_match``, eliminating inter-pass jitter in
    time-series stacks and animations.

    Parameters
    ----------
    items : list[LoadedItem]
        Clipped items (one per pass).
    ref_item : LoadedItem
        The item whose grid all others are matched to (typically the
        first pass).

    Returns
    -------
    list[LoadedItem]
        Items with all DataArrays aligned to the reference grid.
    """
    # Pick a reference DataArray (first asset of the reference item)
    ref_asset_name = next(iter(ref_item.data))
    ref_da = ref_item.data[ref_asset_name]

    for item in items:
        if item is ref_item:
            continue
        for asset_name, da in item.data.items():
            # Only reproject if shapes or transforms actually differ
            if da.shape == ref_da.shape:
                try:
                    if da.rio.transform() == ref_da.rio.transform():
                        continue
                except Exception:
                    pass
            try:
                item.data[asset_name] = da.rio.reproject_match(ref_da)
            except Exception as exc:
                logger.warning(
                    "reproject_match failed for %s asset %s: %s",
                    item.id, asset_name, exc,
                )
    return items


def mosaic_items(items: List[LoadedItem], bbox=None) -> List[LoadedItem]:
    """Mosaic burst-level items into one item per satellite pass.

    Items are grouped by track number and acquisition date. Within
    each group the per-asset DataArrays are spatially merged using
    ``rioxarray.merge.merge_arrays``.

    Bursts are merged on their native pixel grid *before* applying any
    bounding-box clip.  This avoids the 1‒2 pixel misalignment that
    occurs when each burst is independently snapped to a box boundary
    during loading.

    Parameters
    ----------
    items : list[LoadedItem]
        Burst-level loaded items (as returned by :func:`load_items`).
    bbox : BoundingBox, optional
        If provided, clip each merged mosaic to this bounding box after
        merging.  Applied to single-burst items too.

    Returns
    -------
    list[LoadedItem]
        One :class:`LoadedItem` per pass, sorted by datetime.
    """
    groups = group_by_pass(items)

    if not groups:
        return items

    result: List[LoadedItem] = []

    for (track, date_str), group in sorted(groups.items()):
        if len(group) == 1:
            item = group[0]
            if bbox is not None:
                item = _clip_item_to_bbox(item, bbox)
            result.append(item)
            continue

        # Sort bursts by acquisition time for deterministic ordering
        group.sort(key=lambda x: x.datetime)

        # Collect asset names across all items in the group
        asset_names = set()
        for item in group:
            asset_names.update(item.data.keys())

        merged_data: Dict[str, object] = {}
        for asset_name in sorted(asset_names):
            arrays = [
                item.data[asset_name]
                for item in group
                if asset_name in item.data
            ]
            if len(arrays) == 1:
                merged_data[asset_name] = arrays[0]
            else:
                merged_data[asset_name] = _merge_arrays(arrays)
                # Release individual burst arrays to free memory
                for arr in arrays:
                    try:
                        arr.close()
                    except Exception:
                        pass
                del arrays

        # Use metadata from the first burst
        ref = group[0]
        burst_ids = [item.id for item in group]
        mosaic_id = f"T{track:03d}_{date_str}_{len(group)}bursts"

        merged_item = LoadedItem(
            id=mosaic_id,
            datetime=ref.datetime,
            platform=ref.platform,
            orbit_direction=ref.orbit_direction,
            data=merged_data,
            crs=ref.crs,
            pixel_size_m=ref.pixel_size_m,
        )

        # Clip merged mosaic to the requested AOI now that all bursts share
        # the same native pixel grid (no individual-burst boundary snapping).
        if bbox is not None:
            merged_item = _clip_item_to_bbox(merged_item, bbox)

        n_bursts = len(group)
        orb = (ref.orbit_direction or "?")[:3].upper()
        print(
            f"  Mosaicked T{track:03d} {orb} {date_str}: "
            f"{n_bursts} bursts → "
            f"{next(iter(merged_item.data.values())).shape}"
        )
        logger.info(
            "Mosaicked %d bursts for track %d on %s: %s",
            n_bursts, track, date_str, burst_ids,
        )

        result.append(merged_item)

    result.sort(key=lambda x: x.datetime)

    # Snap all passes to the first pass's pixel grid so that every date
    # in the time-series shares exactly the same spatial extent and
    # resolution — eliminates ±1 pixel cross-pass jitter.
    if len(result) > 1:
        result = _snap_to_reference(result, result[0])

    return result
