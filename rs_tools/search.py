"""Unified search interface across all archive connectors."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from rs_tools.archives.base import BaseArchive
from rs_tools.archives.cdse import CDSEArchive
from rs_tools.archives.nasa import NASAArchive
from rs_tools.archives.terrascope import TerrascopeArchive
from rs_tools.config import SearchConfig

logger = logging.getLogger(__name__)

_ARCHIVE_REGISTRY: Dict[str, type] = {
    "cdse": CDSEArchive,
    "nasa": NASAArchive,
    "terrascope": TerrascopeArchive,
}


def get_archive(name: str) -> BaseArchive:
    """Instantiate an archive connector by name.

    Parameters
    ----------
    name : str
        One of ``"cdse"``, ``"nasa"``, ``"terrascope"``.

    Returns
    -------
    BaseArchive
        A ready-to-use archive connector.
    """
    key = name.lower()
    if key not in _ARCHIVE_REGISTRY:
        raise ValueError(
            f"Unknown archive {name!r}. "
            f"Available: {sorted(_ARCHIVE_REGISTRY)}"
        )
    return _ARCHIVE_REGISTRY[key]()


def search_archive(
    archive: Union[str, BaseArchive],
    config: SearchConfig,
) -> List[Dict[str, Any]]:
    """Run a search against a single archive.

    Parameters
    ----------
    archive : str | BaseArchive
        Archive name or an already-instantiated connector.
    config : SearchConfig
        Uniform search parameters.

    Returns
    -------
    list[dict]
        Item metadata dictionaries.
    """
    if isinstance(archive, str):
        archive = get_archive(archive)
    return archive.search(config)


def search_with_fallback(
    config: SearchConfig,
    archives: List[str],
) -> tuple:
    """Try archives in priority order, returning the first successful result.

    Parameters
    ----------
    config : SearchConfig
        Uniform search parameters.
    archives : list[str]
        Archive names in priority order (first = preferred).

    Returns
    -------
    tuple[str, list[dict]]
        ``(archive_name, items)`` from the first archive that returned
        results without error.  If all archives fail or return empty,
        returns ``(last_archive, [])``.
    """
    for name in archives:
        try:
            items = search_archive(name, config)
            if items:
                logger.info(
                    "search_with_fallback: %s returned %d items", name, len(items)
                )
                return name, items
            logger.info("search_with_fallback: %s returned 0 items, trying next", name)
        except Exception:
            logger.warning(
                "search_with_fallback: %s failed, trying next", name, exc_info=True
            )
    return archives[-1] if archives else "unknown", []


def search_all(
    config: SearchConfig,
    archives: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Search multiple archives with the same query.

    Parameters
    ----------
    config : SearchConfig
        Uniform search parameters.
    archives : list[str] | None
        Archive names to query.  Defaults to all registered archives.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of archive name to its search results.
    """
    if archives is None:
        archives = list(_ARCHIVE_REGISTRY)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for name in archives:
        try:
            results[name] = search_archive(name, config)
        except Exception:
            logger.exception("Search failed for archive %s", name)
            results[name] = []
    return results
