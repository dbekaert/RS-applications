"""Connector for the Copernicus Data Space Ecosystem (CDSE).

Supports:
* **STAC** catalogue search for Sentinel and other products that are
  indexed in the CDSE STAC API.
* **OData** search for CLMS (Copernicus Land Monitoring Service) global
  products that are *not* in STAC but are available via the CDSE OData
  catalogue and stored on S3 object storage.
* Efficient streaming of Cloud-Optimized GeoTIFFs via GDAL ``/vsis3/``
  with automatic temporary-credential management.
"""

from __future__ import annotations

import atexit
import logging
import netrc
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from pystac_client import Client

from rs_tools.archives.base import BaseArchive
from rs_tools.config import BoundingBox, SearchConfig

logger = logging.getLogger(__name__)

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)
CDSE_S3_KEYS_URL = "https://s3-keys-manager.cloudferro.com/api/user/credentials"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

# Module-level state for temp-credential lifecycle
_temp_s3_creds: Optional[Dict[str, str]] = None
_temp_s3_token: Optional[str] = None


# -------------------------------------------------------------------
# GDAL configuration
# -------------------------------------------------------------------

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


def configure_gdal_cdse_s3() -> None:
    """Configure GDAL ``/vsis3/`` for CDSE EODATA object storage.

    Obtains temporary S3 credentials from the CDSE keys-manager API
    using ``.netrc`` credentials and sets the required GDAL/AWS
    environment variables.  Credentials are cleaned up at process exit.
    """
    global _temp_s3_creds, _temp_s3_token  # noqa: PLW0603

    # Already configured?
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_S3_ENDPOINT") == CDSE_S3_ENDPOINT:
        logger.debug("CDSE S3 environment already configured")
        return

    configure_gdal_cdse()  # base caching settings

    token = _get_cdse_token()
    if token is None:
        raise RuntimeError(
            "Cannot obtain CDSE OAuth2 token.  Add your CDSE credentials "
            "to ~/.netrc:\n"
            "  machine dataspace.copernicus.eu\n"
            "      login YOUR_EMAIL\n"
            "      password YOUR_PASSWORD"
        )

    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(CDSE_S3_KEYS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    creds = resp.json()
    logger.info("Temporary CDSE S3 credentials created (access_id=%s…)",
                creds["access_id"][:8])

    _temp_s3_creds = creds
    _temp_s3_token = token

    os.environ["AWS_S3_ENDPOINT"] = CDSE_S3_ENDPOINT
    os.environ["AWS_ACCESS_KEY_ID"] = creds["access_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds["secret"]
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["AWS_HTTPS"] = "YES"

    # Allow a moment for the key pair to propagate
    time.sleep(2)

    atexit.register(_cleanup_temp_s3_creds)


def _get_cdse_token() -> Optional[str]:
    """Obtain a CDSE OAuth2 bearer token from ``.netrc`` credentials."""
    try:
        nrc = netrc.netrc()
    except FileNotFoundError:
        logger.warning("~/.netrc not found")
        return None

    for host in ("dataspace.copernicus.eu", "identity.dataspace.copernicus.eu"):
        entry = nrc.authenticators(host)
        if entry:
            username, _, password = entry
            resp = requests.post(
                CDSE_TOKEN_URL,
                data={
                    "client_id": "cdse-public",
                    "grant_type": "password",
                    "username": username,
                    "password": password,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["access_token"]
            logger.warning("CDSE token request failed: %d", resp.status_code)
    return None


def _cleanup_temp_s3_creds() -> None:
    """Delete temporary S3 credentials on process exit."""
    global _temp_s3_creds, _temp_s3_token  # noqa: PLW0603
    if _temp_s3_creds is None or _temp_s3_token is None:
        return
    try:
        access_id = _temp_s3_creds["access_id"]
        headers = {"Authorization": f"Bearer {_temp_s3_token}"}
        requests.delete(
            f"{CDSE_S3_KEYS_URL}/access_id/{access_id}",
            headers=headers,
            timeout=10,
        )
        logger.info("Temporary CDSE S3 credentials deleted (%s…)", access_id[:8])
    except Exception:
        logger.debug("Failed to delete temp S3 credentials", exc_info=True)
    _temp_s3_creds = None
    _temp_s3_token = None


# -------------------------------------------------------------------
# OData helpers for CLMS products
# -------------------------------------------------------------------

def _is_clms_collection(collection: str) -> bool:
    """Return *True* if *collection* is a CLMS S3 path segment."""
    return "/" in collection


def _clms_cog_assets(product_name: str, s3_path: str) -> Dict[str, Any]:
    """Derive ``/vsis3/`` asset HREFs from a CLMS OData product.

    Parameters
    ----------
    product_name : str
        e.g. ``"c_gls_NDVI300_202001110000_GLOBE_OLCI_V3.0.1_cog"``
    s3_path : str
        e.g. ``"/eodata/CLMS/.../c_gls_NDVI300_…_cog"``

    Returns
    -------
    dict
        STAC-style assets dict with at least a ``"data"`` key whose
        ``href`` is a ``/vsis3/`` URL pointing to the primary band COG.
    """
    # Strip _cog suffix to get the base name used in individual TIFFs
    base = product_name
    if base.endswith("_cog"):
        base = base[:-4]

    # Product code is the 3rd underscore-separated part:  c_gls_NDVI300_…
    parts = base.split("_")
    product_code = parts[2] if len(parts) >= 3 else ""

    # Primary band name: strip trailing digits from the product code
    primary_band = re.sub(r"\d+$", "", product_code)

    # S3 path → /vsis3/ URL:  /eodata/CLMS/… → /vsis3/eodata/CLMS/…
    vsis3_base = "/vsis3" + s3_path

    if primary_band:
        tiff_name = base.replace(product_code, f"{product_code}-{primary_band}") + ".tiff"
    else:
        tiff_name = base + ".tiff"

    return {
        "data": {
            "href": f"{vsis3_base}/{tiff_name}",
            "type": "image/tiff",
            "title": primary_band or "data",
        },
    }


def _odata_product_to_item(product: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single OData product dict to a STAC-like item dict."""
    name = product["Name"]
    s3_path = product["S3Path"]
    content_date = product["ContentDate"]
    geo = product.get("GeoFootprint")

    assets = _clms_cog_assets(name, s3_path)

    bbox = None
    if geo and "coordinates" in geo:
        coords = geo["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox = [min(lons), min(lats), max(lons), max(lats)]

    return {
        "id": name,
        "type": "Feature",
        "properties": {
            "datetime": content_date["Start"],
            "platform": "CLMS",
            "start_datetime": content_date["Start"],
            "end_datetime": content_date["End"],
        },
        "assets": assets,
        "geometry": geo,
        "bbox": bbox,
    }


# -------------------------------------------------------------------
# Archive connector
# -------------------------------------------------------------------

class CDSEArchive(BaseArchive):
    """Interface to the Copernicus Data Space Ecosystem catalogue.

    Sentinel / OPERA products are searched via STAC.  CLMS global
    products (whose collection identifiers contain ``/``) are searched
    via the OData API.
    """

    name = "cdse"

    def __init__(self, stac_url: str = CDSE_STAC_URL) -> None:
        self._url = stac_url
        self._client: Optional[Client] = None

    def _stac_client(self) -> Client:
        if self._client is None:
            self._client = Client.open(self._url)
        return self._client

    def list_collections(self) -> List[str]:
        """Return available CDSE STAC collection identifiers."""
        return [c.id for c in self._stac_client().get_collections()]

    # ---------------------------------------------------------------
    # Unified search entry-point
    # ---------------------------------------------------------------

    def search(self, config: SearchConfig) -> List[Dict[str, Any]]:
        """Search the CDSE catalogue.

        CLMS collections (identified by a ``/`` in their name) are
        searched via OData; all other collections use the STAC API.

        Parameters
        ----------
        config : SearchConfig
            Uniform search parameters.

        Returns
        -------
        list[dict]
            Item metadata dictionaries.
        """
        clms_cols = [c for c in (config.collections or []) if _is_clms_collection(c)]
        stac_cols = [c for c in (config.collections or []) if not _is_clms_collection(c)]

        items: List[Dict[str, Any]] = []

        if clms_cols:
            items.extend(self._search_odata_clms(config, clms_cols))
        if stac_cols or not config.collections:
            items.extend(self._search_stac(config, stac_cols or config.collections))

        return items

    # ---------------------------------------------------------------
    # STAC search (Sentinel, OPERA, …)
    # ---------------------------------------------------------------

    def _search_stac(
        self,
        config: SearchConfig,
        collections: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Search CDSE STAC catalogue."""
        search_kwargs: Dict[str, Any] = {
            "bbox": config.bbox.as_list(),
            "datetime": config.date_range_str,
            "max_items": config.limit,
        }
        if collections:
            search_kwargs["collections"] = collections

        results = self._stac_client().search(**search_kwargs)
        items = [item.to_dict() for item in results.items()]
        logger.info("CDSE STAC search returned %d items.", len(items))
        return items

    # ---------------------------------------------------------------
    # OData search (CLMS global products)
    # ---------------------------------------------------------------

    def _search_odata_clms(
        self,
        config: SearchConfig,
        collection_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """Search CLMS products via the OData API.

        Parameters
        ----------
        config : SearchConfig
            Search parameters (dates, bbox, limit).
        collection_paths : list[str]
            CLMS S3 path segments, e.g.
            ``["bio-geophysical/vegetation_indices/ndvi_global_300m_10daily_v3"]``.

        Returns
        -------
        list[dict]
            STAC-like item dictionaries with ``/vsis3/`` asset HREFs.
        """
        all_items: List[Dict[str, Any]] = []
        for col_path in collection_paths:
            dataset_id = col_path.rsplit("/", 1)[-1]
            products = self._odata_query(
                dataset_id,
                start=config.start_date.isoformat(),
                end=config.end_date.isoformat(),
                limit=config.limit,
            )
            all_items.extend(_odata_product_to_item(p) for p in products)

        logger.info("CDSE OData search returned %d CLMS items.", len(all_items))
        return all_items

    @staticmethod
    def _odata_query(
        dataset_id: str,
        start: str,
        end: str,
        limit: int = 150,
    ) -> List[Dict[str, Any]]:
        """Page through the OData API for a single CLMS dataset.

        Filters for COG products only (``ContentType eq
        'application/tiff'``).
        """
        filters = [
            "Collection/Name eq 'CLMS'",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'datasetIdentifier' and "
                f"att/OData.CSC.StringAttribute/Value eq '{dataset_id}')"
            ),
            "endswith(Name,'_cog')",
            f"ContentDate/Start ge {start}T00:00:00.000Z",
            f"ContentDate/Start le {end}T23:59:59.999Z",
        ]

        filter_str = " and ".join(filters)
        page_size = min(limit, 1000)
        products: List[Dict[str, Any]] = []
        skip = 0

        while len(products) < limit:
            params = {
                "$filter": filter_str,
                "$orderby": "ContentDate/Start asc",
                "$top": page_size,
                "$skip": skip,
                "$select": "Id,Name,S3Path,ContentDate,GeoFootprint",
            }
            resp = requests.get(
                f"{CDSE_ODATA_URL}/Products", params=params, timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("value", [])
            if not batch:
                break
            products.extend(batch)
            skip += len(batch)
            if "@odata.nextLink" not in data:
                break

        return products[:limit]
