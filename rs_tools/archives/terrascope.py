"""Connector for the Terrascope STAC API."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from pystac_client import Client

from rs_tools.archives.base import BaseArchive
from rs_tools.config import SearchConfig

logger = logging.getLogger(__name__)

TERRASCOPE_STAC_URL = "https://stac.terrascope.be/"

# Retry settings for transient network failures
_MAX_RETRIES = 4
_BACKOFF_BASE = 5  # seconds; doubles each retry: 5, 10, 20, 40


class TerrascopeArchive(BaseArchive):
    """Interface to the Terrascope STAC catalogue."""

    name = "terrascope"

    def __init__(self, stac_url: str = TERRASCOPE_STAC_URL) -> None:
        self._url = stac_url
        self._client = Client.open(self._url)

    def list_collections(self) -> List[str]:
        """Return available Terrascope collection identifiers."""
        return [c.id for c in self._client.get_collections()]

    def search(self, config: SearchConfig) -> List[Dict[str, Any]]:
        """Search Terrascope STAC catalogue.

        Retries up to ``_MAX_RETRIES`` times on transient network
        errors (connection resets, timeouts, server errors).

        Parameters
        ----------
        config : SearchConfig
            Uniform search parameters.

        Returns
        -------
        list[dict]
            Item metadata dictionaries.
        """
        search_kwargs: Dict[str, Any] = {
            "bbox": config.bbox.as_list(),
            "datetime": config.date_range_str,
            "max_items": config.limit,
        }
        if config.collections:
            search_kwargs["collections"] = config.collections

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                results = self._client.search(**search_kwargs)
                items = [item.to_dict() for item in results.items()]
                logger.info("Terrascope search returned %d items.", len(items))
                return items
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "Terrascope search attempt %d/%d failed (%s), "
                        "retrying in %ds …",
                        attempt + 1, _MAX_RETRIES + 1, exc, wait,
                    )
                    time.sleep(wait)
                    # Re-create the client in case the connection is stale
                    try:
                        self._client = Client.open(self._url)
                    except Exception:
                        pass

        raise RuntimeError(
            f"Terrascope search failed after {_MAX_RETRIES + 1} attempts"
        ) from last_exc
