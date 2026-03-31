"""Known dataset catalog and product registry.

Centralises metadata for datasets that the toolkit knows how to locate
across the supported archives.  Each entry maps a human-friendly name to
the collection identifiers understood by each archive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DatasetInfo:
    """Metadata for a single known dataset / product.

    Attributes
    ----------
    name : str
        Human-readable display name.
    short_name : str
        Compact identifier (e.g. ``"NDVI"``).
    description : str
        One-line description.
    archive_collections : dict[str, list[str]]
        Mapping ``archive_name -> [collection_id, ...]``.
    version : str | None
        Product version string if applicable.
    temporal_resolution : str | None
        E.g. ``"10-daily"``, ``"monthly"``.
    spatial_resolution : str | None
        E.g. ``"300 m"``, ``"1 km"``.
    is_global : bool
        If *True*, the product is a global composite (e.g. CGLOPS
        dekadal/monthly products).  Global products are never routed
        through OPERA-specific grouping or pass-parsing logic.
    tags : list[str]
        Arbitrary tags for filtering.
    """

    name: str
    short_name: str
    description: str
    archive_collections: Dict[str, List[str]] = field(default_factory=dict)
    version: Optional[str] = None
    temporal_resolution: Optional[str] = None
    spatial_resolution: Optional[str] = None
    is_global: bool = False
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Global catalog
# ---------------------------------------------------------------------------

_CATALOG: Dict[str, DatasetInfo] = {}


def register(dataset: DatasetInfo) -> None:
    """Register a dataset in the global catalog."""
    _CATALOG[dataset.short_name] = dataset


def get(short_name: str) -> DatasetInfo:
    """Look up a dataset by its short name."""
    if short_name not in _CATALOG:
        raise KeyError(
            f"Unknown dataset {short_name!r}. "
            f"Available: {sorted(_CATALOG)}"
        )
    return _CATALOG[short_name]


def list_datasets(tag: Optional[str] = None) -> List[DatasetInfo]:
    """Return all registered datasets, optionally filtered by tag."""
    datasets = list(_CATALOG.values())
    if tag:
        datasets = [d for d in datasets if tag in d.tags]
    return datasets


# ---------------------------------------------------------------------------
# CLMS Bio-Geophysical Products on CDSE  (CSV catalogue + S3 storage)
#
# These products are NOT in the CDSE STAC API.  They are hosted on
# CDSE S3 object storage (s3://EODATA/CLMS/...) with CSV-based
# catalogues at https://csv.dataspace.copernicus.eu/CLMS/...
# The ``archive_collections`` values are the S3 path segments under
# ``s3://EODATA/CLMS/`` that uniquely identify each product.
# ---------------------------------------------------------------------------

# --- Vegetation Indices ------------------------------------------------

register(DatasetInfo(
    name="CLMS NDVI v3",
    short_name="CLMS_NDVI_V3",
    description="Normalized Difference Vegetation Index (CGLOPS, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/vegetation_indices/ndvi_global_300m_10daily_v3"],
    },
    version="3",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "vegetation", "ndvi"],
))

# --- Vegetation Properties ---------------------------------------------

register(DatasetInfo(
    name="CLMS LAI v2",
    short_name="CLMS_LAI_V2",
    description="Leaf Area Index (CGLOPS v2, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/vegetation_properties/lai_global_300m_10daily_v2"],
    },
    version="2",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "vegetation", "lai"],
))

register(DatasetInfo(
    name="CLMS FAPAR v2",
    short_name="CLMS_FAPAR_V2",
    description="Fraction of Absorbed Photosynthetically Active Radiation (CGLOPS v2, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/vegetation_properties/fapar_global_300m_10daily_v2"],
    },
    version="2",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "vegetation", "fapar"],
))

register(DatasetInfo(
    name="CLMS FCOVER v2",
    short_name="CLMS_FCOVER_V2",
    description="Fraction of Vegetation Cover (CGLOPS v2, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/vegetation_properties/fcover_global_300m_10daily_v2"],
    },
    version="2",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "vegetation", "fcover"],
))

# --- Carbon / Primary Production ---------------------------------------

register(DatasetInfo(
    name="CLMS GPP v2",
    short_name="CLMS_GPP_V2",
    description="Gross Primary Production (CGLOPS v2, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/net-gross_primary_production/gpp_global_300m_10daily_v2"],
    },
    version="2",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "carbon", "gpp"],
))

register(DatasetInfo(
    name="CLMS NPP v2",
    short_name="CLMS_NPP_V2",
    description="Net Primary Production (CGLOPS v2, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/net-gross_primary_production/npp_global_300m_10daily_v2"],
    },
    version="2",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "carbon", "npp"],
))

# --- Evapotranspiration ------------------------------------------------

register(DatasetInfo(
    name="CLMS ETA v1",
    short_name="CLMS_ETA_V1",
    description="Actual Evapotranspiration (CGLOPS v1, 10-daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/evapotranspiration/eta_global_300m_10daily_v1"],
    },
    version="1",
    temporal_resolution="10-daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "evapotranspiration", "eta"],
))

register(DatasetInfo(
    name="CLMS Sensible Heat Flux v1",
    short_name="CLMS_HF_V1",
    description="Sensible Heat Flux (CGLOPS v1, daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/evapotranspiration/hf_global_300m_daily_v1"],
    },
    version="1",
    temporal_resolution="daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "evapotranspiration", "hf"],
))

# --- Burnt Area --------------------------------------------------------

register(DatasetInfo(
    name="CLMS Burnt Area v4 daily",
    short_name="CLMS_BA_V4_DAILY",
    description="Burnt Area (CGLOPS v4, daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/burnt_area/ba_global_300m_daily_v4"],
    },
    version="4",
    temporal_resolution="daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "fire", "burnt_area"],
))

register(DatasetInfo(
    name="CLMS Burnt Area v4 monthly",
    short_name="CLMS_BA_V4_MONTHLY",
    description="Burnt Area (CGLOPS v4, monthly, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/burnt_area/ba_global_300m_monthly_v4"],
    },
    version="4",
    temporal_resolution="monthly",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "fire", "burnt_area"],
))

# --- Top-of-Canopy Reflectance ----------------------------------------

register(DatasetInfo(
    name="CLMS TOC Reflectance v2",
    short_name="CLMS_TOC_V2",
    description="Top-of-Canopy Reflectance (CGLOPS v2, daily, 300 m).",
    archive_collections={
        "cdse": ["bio-geophysical/top-of-canopy_reflectances/toc_global_300m_daily_v2"],
    },
    version="2",
    temporal_resolution="daily",
    spatial_resolution="300 m",
    is_global=True,
    tags=["biophysical", "clms", "reflectance", "toc"],
))

# --- Soil Water Index --------------------------------------------------

register(DatasetInfo(
    name="CLMS Soil Water Index v4",
    short_name="CLMS_SWI_V4",
    description="Soil Water Index (CGLOPS v4, daily, 12.5 km).",
    archive_collections={
        "cdse": ["bio-geophysical/soil_water_index/swi_global_12.5km_daily_v4"],
    },
    version="4",
    temporal_resolution="daily",
    spatial_resolution="12.5 km",
    is_global=True,
    tags=["biophysical", "clms", "soil", "swi"],
))

# ---------------------------------------------------------------------------
# OPERA SAR Products  (Terrascope STAC / NASA ASF)
#
# OPERA products are available from both Terrascope (STAC) and NASA's
# ASF DAAC (CMR / HTTPS / S3).  The ``archive_collections`` values for
# NASA are the ASF dataset short names used by asf_search / CMR.
# ---------------------------------------------------------------------------

register(DatasetInfo(
    name="OPERA RTC-S1",
    short_name="OPERA_RTC_S1",
    description="OPERA Radiometric Terrain-Corrected SAR backscatter from Sentinel-1.",
    archive_collections={
        "terrascope": ["opera-s1-rtc-v1"],
        "nasa": ["OPERA_L2_RTC-S1_V1_1"],
    },
    version="1",
    temporal_resolution="6-12 days",
    spatial_resolution="30 m",
    tags=["sar", "opera", "rtc", "sentinel-1", "backscatter"],
))

register(DatasetInfo(
    name="OPERA RTC-S1 Static",
    short_name="OPERA_RTC_S1_STATIC",
    description="OPERA RTC-S1 static layers (layover/shadow mask, incidence angle, etc.).",
    archive_collections={
        "terrascope": ["opera-s1-rtc-static-v1"],
        "nasa": ["OPERA_L2_RTC-S1-STATIC_V1"],
    },
    version="1",
    temporal_resolution=None,
    spatial_resolution="30 m",
    tags=["sar", "opera", "rtc", "sentinel-1", "static"],
))

# ---------------------------------------------------------------------------
# ARIA InSAR Products  (NASA ASF, future: Terrascope STAC)
# ---------------------------------------------------------------------------

register(DatasetInfo(
    name="ARIA S1 GUNW",
    short_name="ARIA_S1_GUNW",
    description="ARIA Geocoded Unwrapped Interferograms from Sentinel-1.",
    archive_collections={
        "nasa": ["ARIA_S1_GUNW"],
    },
    version="1",
    temporal_resolution="6-12 days",
    spatial_resolution="90 m",
    tags=["sar", "aria", "insar", "sentinel-1", "gunw", "interferogram"],
))
