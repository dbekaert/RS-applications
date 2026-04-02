"""Official CLMS colour ramps for the Copernicus Land Monitoring Service.

Each colour ramp is extracted from the official SentinelHub evalscripts
published by Copernicus (eu-cdse/sentinel-hub-custom-scripts).  Values
are given in **physical units** (not raw digital numbers).

The colormaps are registered with matplotlib under names prefixed with
``clms_`` so they can be passed as plain strings to any function that
accepts a matplotlib colormap name, or used directly as ``Colormap``
objects.

Source
------
https://github.com/eu-cdse/sentinel-hub-custom-scripts/tree/main/clms
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_cmap(name: str, anchors: list, vmin: float, vmax: float) -> LinearSegmentedColormap:
    """Build a ``LinearSegmentedColormap`` from (value, (R, G, B)) anchors.

    Parameters
    ----------
    name : str
        Name to register in matplotlib.
    anchors : list[tuple[float, tuple[int, int, int]]]
        Ordered list of ``(physical_value, (R, G, B))`` with RGB in 0-255.
    vmin, vmax : float
        Physical value extent for the ramp.  Anchors are normalised
        into [0, 1] using these bounds.
    """
    positions = [(v - vmin) / (vmax - vmin) for v, _ in anchors]
    colours = [(r / 255, g / 255, b / 255) for _, (r, g, b) in anchors]
    positions = [max(0.0, min(1.0, p)) for p in positions]

    cdict: dict = {"red": [], "green": [], "blue": []}
    for pos, (r, g, b) in zip(positions, colours):
        cdict["red"].append((pos, r, r))
        cdict["green"].append((pos, g, g))
        cdict["blue"].append((pos, b, b))

    cmap = LinearSegmentedColormap(name, segmentdata=cdict, N=256)
    return cmap


# ===================================================================
# NDVI  (physical range: -0.08 .. 0.92)
# Source: ndvi_global_300m_10daily_v3/scripts/ndvi.js
# The evalscript applies the color ramp to raw DN (0-250); here the
# DN anchors are converted to physical values via val = DN/250 - 0.08.
# ===================================================================
_NDVI_ANCHORS = [
    (-0.08, (140, 92, 8)),
    (0.00,  (142, 95, 8)),
    (0.10,  (197, 173, 19)),
    (0.20,  (255, 255, 30)),
    (0.30,  (218, 232, 25)),
    (0.40,  (182, 210, 21)),
    (0.50,  (145, 188, 17)),
    (0.60,  (109, 166, 12)),
    (0.70,  (72, 144, 8)),
    (0.80,  (36, 122, 4)),
    (0.92,  (0, 100, 0)),
]

NDVI_VMIN, NDVI_VMAX = -0.08, 0.92
CLMS_NDVI = _build_cmap("clms_ndvi", _NDVI_ANCHORS, NDVI_VMIN, NDVI_VMAX)


# ===================================================================
# LAI  (physical range: 0 .. 7  m²/m²)
# Source: lai_global_300m_10daily_v2/scripts/lai.js
# ===================================================================
_LAI_ANCHORS = [
    (0, (140, 92, 8)),
    (1, (197, 173, 4)),
    (2, (255, 255, 0)),
    (3, (127, 227, 0)),
    (4, (0, 200, 0)),
    (5, (0, 166, 0)),
    (6, (0, 133, 0)),
    (7, (0, 100, 0)),
]

LAI_VMIN, LAI_VMAX = 0, 7
CLMS_LAI = _build_cmap("clms_lai", _LAI_ANCHORS, LAI_VMIN, LAI_VMAX)


# ===================================================================
# FAPAR  (physical range: 0 .. 0.94)
# Source: fapar_global_300m_10daily_v2/scripts/fapar.js
# ===================================================================
_FAPAR_ANCHORS = [
    (0.00, (140, 92, 8)),
    (0.10, (174, 141, 14)),
    (0.20, (209, 190, 21)),
    (0.30, (243, 239, 27)),
    (0.40, (229, 239, 26)),
    (0.50, (190, 216, 22)),
    (0.60, (152, 192, 17)),
    (0.70, (114, 169, 13)),
    (0.80, (76, 146, 8)),
    (0.90, (38, 123, 4)),
    (0.94, (23, 114, 2)),
]

FAPAR_VMIN, FAPAR_VMAX = 0.0, 0.94
CLMS_FAPAR = _build_cmap("clms_fapar", _FAPAR_ANCHORS, FAPAR_VMIN, FAPAR_VMAX)


# ===================================================================
# FCOVER  (physical range: 0 .. 0.94)
# Source: fcover_global_300m_10daily_v2/scripts/FCOVER.js
# Same ramp as FAPAR.
# ===================================================================
_FCOVER_ANCHORS = _FAPAR_ANCHORS  # identical colour table

FCOVER_VMIN, FCOVER_VMAX = 0.0, 0.94
CLMS_FCOVER = _build_cmap("clms_fcover", _FCOVER_ANCHORS, FCOVER_VMIN, FCOVER_VMAX)


# ===================================================================
# GPP  (physical range: 0 .. 30  gC/m²/day)
# Source: gpp_global_300m_10daily_v2/scripts/gpp.js
# ===================================================================
_GPP_ANCHORS = [
    (0,  (115, 0, 0)),
    (3,  (218, 140, 0)),
    (6,  (255, 183, 135)),
    (9,  (195, 255, 153)),
    (12, (115, 165, 23)),
    (15, (58, 128, 95)),
    (18, (17, 95, 136)),
    (21, (14, 77, 132)),
    (24, (12, 62, 129)),
    (27, (11, 53, 127)),
    (30, (10, 45, 125)),
]

GPP_VMIN, GPP_VMAX = 0, 30
CLMS_GPP = _build_cmap("clms_gpp", _GPP_ANCHORS, GPP_VMIN, GPP_VMAX)


# ===================================================================
# NPP  (physical range: 0 .. 15  gC/m²/day)
# Source: npp_global_300m_10daily_v2/scripts/npp.js
# Same colour stops as GPP, different value scale.
# ===================================================================
_NPP_ANCHORS = [
    (0.0,  (115, 0, 0)),
    (1.5,  (218, 140, 0)),
    (3.0,  (255, 183, 135)),
    (4.5,  (195, 255, 153)),
    (6.0,  (115, 165, 23)),
    (7.5,  (58, 128, 95)),
    (9.0,  (17, 95, 136)),
    (10.5, (14, 77, 132)),
    (12.0, (12, 62, 129)),
    (13.5, (11, 53, 127)),
    (15.0, (10, 45, 125)),
]

NPP_VMIN, NPP_VMAX = 0, 15
CLMS_NPP = _build_cmap("clms_npp", _NPP_ANCHORS, NPP_VMIN, NPP_VMAX)


# ===================================================================
# SWI  (Soil Water Index, physical range: 0 .. 100  %)
# Source: swi_global_12.5km_10daily_v4/scripts/swi001.js
# ===================================================================
_SWI_ANCHORS = [
    (0,   (148, 80, 23)),
    (10,  (172, 118, 47)),
    (20,  (196, 156, 71)),
    (30,  (220, 194, 96)),
    (40,  (245, 233, 121)),
    (50,  (183, 209, 173)),
    (60,  (121, 185, 225)),
    (70,  (97, 152, 203)),
    (80,  (74, 120, 182)),
    (90,  (50, 87, 160)),
    (100, (27, 55, 139)),
]

SWI_VMIN, SWI_VMAX = 0, 100
CLMS_SWI = _build_cmap("clms_swi", _SWI_ANCHORS, SWI_VMIN, SWI_VMAX)


# ===================================================================
# ETA  (Actual Evapotranspiration, physical range: 0 .. 10  mm/day)
# Source: eta_global_300m_10daily_v1/scripts/ET-ENSEMBLE.js
# ===================================================================
_ETA_ANCHORS = [
    (0.0,  (241, 238, 246)),
    (2.5,  (189, 201, 225)),
    (5.0,  (116, 169, 207)),
    (7.5,  (43, 140, 190)),
    (10.0, (4, 90, 141)),
]

ETA_VMIN, ETA_VMAX = 0, 10
CLMS_ETA = _build_cmap("clms_eta", _ETA_ANCHORS, ETA_VMIN, ETA_VMAX)


# ===================================================================
# BA  (Burnt Area — Burned Fraction, physical range: 0 .. 1)
# Source: ba_global_300m_monthly_v4/scripts/burned_fraction.js
# (-1 anchor is skipped; 0..1 range used for the colour ramp.)
# ===================================================================
_BA_ANCHORS = [
    (0.000, (252, 253, 191)),
    (0.053, (253, 229, 166)),
    (0.105, (254, 204, 143)),
    (0.158, (254, 179, 123)),
    (0.211, (253, 154, 106)),
    (0.263, (250, 129, 95)),
    (0.316, (244, 104, 92)),
    (0.368, (232, 83, 98)),
    (0.421, (214, 68, 109)),
    (0.474, (193, 59, 117)),
    (0.526, (171, 51, 125)),
    (0.579, (149, 44, 129)),
    (0.632, (127, 36, 129)),
    (0.684, (107, 28, 129)),
    (0.737, (85, 20, 125)),
    (0.789, (64, 15, 115)),
    (0.842, (41, 17, 91)),
    (0.895, (22, 15, 58)),
    (0.947, (8, 6, 29)),
    (1.000, (0, 0, 4)),
]

BA_VMIN, BA_VMAX = 0, 1
CLMS_BA = _build_cmap("clms_ba", _BA_ANCHORS, BA_VMIN, BA_VMAX)


# ===================================================================
# Convenience lookup by product short-name
# ===================================================================
CLMS_COLORMAPS = {
    "NDVI":   CLMS_NDVI,
    "LAI":    CLMS_LAI,
    "FAPAR":  CLMS_FAPAR,
    "FCOVER": CLMS_FCOVER,
    "GPP":    CLMS_GPP,
    "NPP":    CLMS_NPP,
    "SWI":    CLMS_SWI,
    "ETA":    CLMS_ETA,
    "BA":     CLMS_BA,
}

CLMS_VMINS = {
    "NDVI":   NDVI_VMIN,
    "LAI":    LAI_VMIN,
    "FAPAR":  FAPAR_VMIN,
    "FCOVER": FCOVER_VMIN,
    "GPP":    GPP_VMIN,
    "NPP":    NPP_VMIN,
    "SWI":    SWI_VMIN,
    "ETA":    ETA_VMIN,
    "BA":     BA_VMIN,
}

CLMS_VMAXS = {
    "NDVI":   NDVI_VMAX,
    "LAI":    LAI_VMAX,
    "FAPAR":  FAPAR_VMAX,
    "FCOVER": FCOVER_VMAX,
    "GPP":    GPP_VMAX,
    "NPP":    NPP_VMAX,
    "SWI":    SWI_VMAX,
    "ETA":    ETA_VMAX,
    "BA":     BA_VMAX,
}
