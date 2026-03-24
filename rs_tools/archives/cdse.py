"""Connector for the Copernicus Data Space Ecosystem (CDSE) STAC API.

Supports both the STAC catalogue search and efficient loading of
Cloud-Optimized GeoTIFFs via ``/vsicurl/`` with bbox subsetting.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from pystac_client import Client

from rs_tools.archives.base import BaseArchive
from rs_tools.config import SearchConfig

logger = logging.getLogger(__name__)

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def configure_gdal_cdse() -> None:
    """Configure GDAL for efficient CDSE COG streaming.

    Sets up ``/vsicurl/`` with block caching and connection
    optimisations.
    """
    settings = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt,.nc",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "67108864",
        "GDAL_CACHEMAX": "256",
    }
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    logger.info("GDAL environment configured for CDSE data")


class CDSEArchive(BaseArchive):
    """Interface to the Copernicus Data Space Ecosystem STAC catalogue."""

    name = "cdse"

    def __init__(self, stac_url: str = CDSE_STAC_URL) -> None:
        self._url = stac_url
        self._client = Client.open(self._url)

    def list_collections(self) -> List[str]:
        """Return available CDSE collection identifiers."""
        return [c.id for c in self._client.get_collections()]

    def search(self, config: SearchConfig) -> List[Dict[str, Any]]:
        """Search CDSE STAC catalogue.

        Asset HREFs in the returned items are wrapped with
        ``/vsicurl/`` so they can be streamed directly by GDAL,
        and clip_box'd to the search bbox — avoiding download of
        the full raster.

        Parameters
        ----------
        config : SearchConfig
            Uniform search parameters.

        Returns
        -------
        list[dict]
            Item metadata dictionaries with GDAL-ready asset HREFs.
        """
        search_kwargs: Dict[str, Any] = {
            "bbox": config.bbox.as_list(),
            "datetime": config.date_range_str,
            "max_items": config.limit,
        }
        if config.collections:
            search_kwargs["collections"] = config.collections

        results = self._client.search(**search_kwargs)
        items = [item.to_dict() for item in results.items()]
        logger.info("CDSE search returned %d items.", len(items))
        return items
