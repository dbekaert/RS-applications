#!/usr/bin/env python3
"""Fetch Oosterweel construction works geometry from OpenStreetMap.

Downloads the full trajectory of the Oosterweel connection works
(tunnels, motorway, construction zones, ramps) from OSM via the
Overpass API and writes them as GeoJSON and optionally as a Shapefile.

Usage:
    python scripts/fetch_oosterweel_geojson.py [--output-dir DIR]

Output files:
    oosterweel_trajectory.geojson   – All features in GeoJSON format
    oosterweel_trajectory.shp       – Same as Shapefile (if geopandas available)
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import urllib.parse
from pathlib import Path

# ── Overpass query ──────────────────────────────────────────────────
# Covers a bounding box around Antwerp-North (left + right bank)
# and fetches three categories of OSM features:
#   1. Ways/relations with "Oosterweel" in their name
#   2. highway=construction within the works area
#   3. landuse=construction features with names relating to the works

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Split into smaller queries to avoid timeout
OVERPASS_QUERIES = [
    # 1. Named Oosterweel features (streets, bridge, campus, island, etc.)
    """\
[out:json][timeout:60];
way["name"~"Oosterweel",i](51.19,4.33,51.46,4.50);
out geom;
""",
    # 2. Scheldetunnel and Kanaaltunnels
    """\
[out:json][timeout:60];
way["name"~"Scheldetunnel|Kanaaltunnel",i](51.19,4.33,51.46,4.50);
out geom;
""",
    # 3. R1 highway construction
    """\
[out:json][timeout:60];
way["highway"="construction"]["ref"="R1"](51.19,4.33,51.46,4.50);
out geom;
""",
    # 4. Named construction zones
    """\
[out:json][timeout:60];
(
  way["landuse"="construction"]["name"~"Oosterweel|Scheldetunnel|Sint-Anna|Antwerpen-West",i]
      (51.19,4.33,51.46,4.50);
  way["highway"="construction"]["name"~"Scheldelaan|Bypass",i]
      (51.19,4.33,51.46,4.50);
);
out geom;
""",
]


# ── Feature classification ──────────────────────────────────────────
def _classify(tags: dict) -> str:
    """Return a human-readable category for an OSM element."""
    name = tags.get("name", "").lower()
    highway = tags.get("highway", "")
    landuse = tags.get("landuse", "")
    construction = tags.get("construction", "")
    tunnel = tags.get("tunnel", "")

    if "scheldetunnel" in name and tunnel == "yes":
        return "Scheldetunnel"
    if "kanaaltunnel" in name and tunnel == "yes":
        return "Kanaaltunnels"
    if "bypass" in name:
        return "Bypass R1"
    if "oosterweelknooppunt" in name and landuse == "construction":
        return "Oosterweelknooppunt (construction zone)"
    if "werf scheldetunnel" in name:
        return "Scheldetunnel construction site"
    if "werf knooppunt" in name:
        return "Junction construction site"
    if "betoncentrale" in name:
        return "Concrete plant (Oosterweelwerken)"
    if "oosterweelcampus" in name:
        return "Oosterweelcampus (Lantis)"
    if "oosterweeleiland" in name:
        return "Oosterweeleiland"
    if "werfweg" in name or "werfbrug" in name:
        return "Construction access road/bridge"
    if highway == "construction" and tunnel == "yes":
        return "Tunnel (under construction)"
    if highway == "construction" and construction == "motorway":
        return "Motorway R1 (under construction)"
    if highway == "construction" and "motorway_link" in construction:
        return "Motorway ramp (under construction)"
    if highway == "construction":
        return "Road under construction"
    if landuse == "construction":
        return "Construction zone"
    if "oosterweelsteenweg" in name:
        return "Oosterweelsteenweg"
    if "oosterweelbrug" in name:
        return "Oosterweelbrug"
    return "Other Oosterweel feature"


def _is_trajectory(tags: dict) -> bool:
    """Keep only features that are part of the actual Oosterweel works."""
    name = tags.get("name", "").lower()
    highway = tags.get("highway", "")
    landuse = tags.get("landuse", "")

    # Tunnel / motorway construction
    if highway == "construction":
        return True
    # Named construction zones (Oosterweelknooppunt, Werf Scheldetunnel, etc.)
    if landuse == "construction" and any(
        kw in name
        for kw in (
            "oosterweel",
            "scheldetunnel",
            "sint-anna",
            "antwerpen-west",
            "betoncentrale",
        )
    ):
        return True
    # Named streets / bridges / campus / island
    if any(
        kw in name
        for kw in ("oosterweelsteenweg", "oosterweelbrug", "oosterweelcampus",
                    "oosterweeleiland", "bypass r1", "scheldetunnel", "kanaaltunnel")
    ):
        return True
    return False


# ── Geometry conversion ─────────────────────────────────────────────
def _way_to_geojson(element: dict) -> dict | None:
    """Convert an Overpass 'way' element to a GeoJSON Feature."""
    geom = element.get("geometry", [])
    tags = element.get("tags", {})
    if not geom or not _is_trajectory(tags):
        return None

    coords = [[pt["lon"], pt["lat"]] for pt in geom]

    # Closed polygon? (landuse areas)
    if coords[0] == coords[-1] and len(coords) > 3:
        geom_type = "Polygon"
        coordinates = [coords]
    else:
        geom_type = "LineString"
        coordinates = coords

    category = _classify(tags)
    props = {
        "osm_id": element["id"],
        "name": tags.get("name", ""),
        "category": category,
        "highway": tags.get("highway", ""),
        "construction": tags.get("construction", ""),
        "tunnel": tags.get("tunnel", ""),
        "landuse": tags.get("landuse", ""),
        "ref": tags.get("ref", ""),
        "operator": tags.get("operator", ""),
        "oneway": tags.get("oneway", ""),
        "layer": tags.get("layer", ""),
    }
    return {
        "type": "Feature",
        "geometry": {"type": geom_type, "coordinates": coordinates},
        "properties": props,
    }


# ── Main logic ──────────────────────────────────────────────────────
def fetch_overpass(query: str) -> dict:
    """Send a query to the Overpass API and return the JSON response."""
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        headers={"User-Agent": "RS-applications/1.0 (Oosterweel analysis)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def fetch_all_elements() -> list[dict]:
    """Run all Overpass sub-queries and merge + deduplicate elements."""
    import time
    all_elements: dict[int, dict] = {}
    for i, query in enumerate(OVERPASS_QUERIES, 1):
        print(f"  Query {i}/{len(OVERPASS_QUERIES)} …", end=" ", flush=True)
        try:
            result = fetch_overpass(query)
            elements = result.get("elements", [])
            new = 0
            for el in elements:
                eid = el.get("id")
                if eid not in all_elements:
                    all_elements[eid] = el
                    new += 1
            print(f"{len(elements)} elements ({new} new)")
        except Exception as exc:
            print(f"FAILED: {exc}")
        if i < len(OVERPASS_QUERIES):
            time.sleep(15)  # Respect Overpass rate limiting
    return list(all_elements.values())


def build_geojson(elements: list[dict]) -> dict:
    """Build a GeoJSON FeatureCollection from Overpass elements."""
    features = []
    for el in elements:
        if el["type"] == "way":
            feat = _way_to_geojson(el)
            if feat is not None:
                features.append(feat)
    return {"type": "FeatureCollection", "features": features}


def write_geojson(fc: dict, path: Path) -> None:
    """Write GeoJSON FeatureCollection to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"  ✓ GeoJSON: {path}  ({len(fc['features'])} features)")


def write_shapefile(fc: dict, path: Path) -> None:
    """Write GeoJSON FeatureCollection as Shapefile (requires geopandas)."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  ⚠ geopandas not installed – skipping Shapefile export")
        return
    gdf = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326")
    gdf.to_file(path, driver="ESRI Shapefile")
    print(f"  ✓ Shapefile: {path}  ({len(gdf)} features)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory for output files (default: output/)",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)

    print("Fetching Oosterweel construction data from OpenStreetMap …")
    elements = fetch_all_elements()
    print(f"  Total unique elements: {len(elements)}")

    fc = build_geojson(elements)

    # Print summary by category
    cats: dict[str, int] = {}
    for f in fc["features"]:
        c = f["properties"]["category"]
        cats[c] = cats.get(c, 0) + 1
    print("\n  Category breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    stem = "oosterweel_trajectory"
    write_geojson(fc, out / f"{stem}.geojson")
    write_shapefile(fc, out / f"{stem}.shp")
    print("\nDone.")


if __name__ == "__main__":
    main()
