#!/usr/bin/env python
"""Download all CLMS data for the CGOPS animation notebooks.

Runs each product download sequentially, skipping passes already on disk.
Uses dekad 1 for all products (matching the notebook settings).

Usage:
    python scripts/download_all_cgops.py
    python scripts/download_all_cgops.py --dekads 1      # explicit dekad 1
    python scripts/download_all_cgops.py --dekads 1 2 3  # all dekads
"""

import argparse
import sys
import time
import traceback

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk


BASE_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS"

# Shared parameters
START = "2020-01-01"
END = "2026-03-01"
LIMIT = 150

# Bounding boxes
BBOX_EUROPE = BoundingBox(west=-10, south=35, east=25, north=60)
BBOX_AMAZON = BoundingBox(west=-70, south=-10, east=-50, north=5)
BBOX_SAHEL = BoundingBox(west=-10, south=10, east=15, north=20)
BBOX_CENTRAL_EU = BoundingBox(west=-5, south=42, east=20, north=55)
BBOX_IBERIA = BoundingBox(west=-10, south=36, east=4, north=44)

# All download tasks: (label, short_name, bbox, output_dir, limit)
TASKS = [
    # breathing_pulse — NDVI over Western Europe
    ("breathing_pulse/NDVI", "CLMS_NDVI_V3", BBOX_EUROPE,
     f"{BASE_DIR}/breathing_pulse", LIMIT),

    # amazon_vs_sahel — NDVI for Amazon and Sahel
    ("amazon_vs_sahel/Amazon", "CLMS_NDVI_V3", BBOX_AMAZON,
     f"{BASE_DIR}/amazon_vs_sahel/amazon", LIMIT),
    ("amazon_vs_sahel/Sahel", "CLMS_NDVI_V3", BBOX_SAHEL,
     f"{BASE_DIR}/amazon_vs_sahel/sahel", LIMIT),

    # carbon_engine — GPP and NPP over Western Europe
    ("carbon_engine/GPP", "CLMS_GPP_V2", BBOX_EUROPE,
     f"{BASE_DIR}/carbon_engine/gpp", LIMIT),
    ("carbon_engine/NPP", "CLMS_NPP_V2", BBOX_EUROPE,
     f"{BASE_DIR}/carbon_engine/npp", LIMIT),

    # drought_pulse — SWI and ETA over Central Europe
    ("drought_pulse/SWI", "CLMS_SWI_V4", BBOX_CENTRAL_EU,
     f"{BASE_DIR}/drought_pulse/swi", LIMIT),
    ("drought_pulse/ETA", "CLMS_ETA_V1", BBOX_CENTRAL_EU,
     f"{BASE_DIR}/drought_pulse/eta", LIMIT),

    # fire_recovery — NDVI and BA over Iberian Peninsula
    ("fire_recovery/NDVI", "CLMS_NDVI_V3", BBOX_IBERIA,
     f"{BASE_DIR}/fire_recovery/ndvi", LIMIT),
    ("fire_recovery/BA", "CLMS_BA_V4_MONTHLY", BBOX_IBERIA,
     f"{BASE_DIR}/fire_recovery/ba", 50),

    # vegetation_anatomy — NDVI, LAI, FAPAR, FCOVER over Western Europe
    ("vegetation_anatomy/NDVI", "CLMS_NDVI_V3", BBOX_EUROPE,
     f"{BASE_DIR}/vegetation_anatomy/ndvi", LIMIT),
    ("vegetation_anatomy/LAI", "CLMS_LAI_V2", BBOX_EUROPE,
     f"{BASE_DIR}/vegetation_anatomy/lai", LIMIT),
    ("vegetation_anatomy/FAPAR", "CLMS_FAPAR_V2", BBOX_EUROPE,
     f"{BASE_DIR}/vegetation_anatomy/fapar", LIMIT),
    ("vegetation_anatomy/FCOVER", "CLMS_FCOVER_V2", BBOX_EUROPE,
     f"{BASE_DIR}/vegetation_anatomy/fcover", LIMIT),

    # multitemporal_rgb — NDVI over Western Europe
    ("multitemporal_rgb/NDVI", "CLMS_NDVI_V3", BBOX_EUROPE,
     f"{BASE_DIR}/multitemporal_rgb", LIMIT),
]


def main():
    parser = argparse.ArgumentParser(description="Download all CGOPS CLMS data")
    parser.add_argument(
        "--dekads", nargs="+", type=int, choices=[1, 2, 3], default=[1],
        help="Dekad(s) to download. Default: [1]",
    )
    args = parser.parse_args()
    dekads = args.dekads

    print(f"{'='*60}")
    print(f"CGOPS bulk data download — dekads={dekads}")
    print(f"{'='*60}\n")

    results = []
    for task in TASKS:
        label, short_name, bbox, output_dir, limit = task

        # Check existing passes
        existing = load_passes_from_disk(output_dir, dekads=dekads)
        if len(existing) >= limit:
            print(f"[SKIP] {label}: already have {len(existing)} passes on disk\n")
            results.append((label, "skipped", len(existing)))
            continue

        print(f"[{len(results)+1}/{len(TASKS)}] {label}: {short_name}")
        print(f"  bbox={bbox}, output={output_dir}")
        print(f"  existing={len(existing)}, target={limit}, dekads={dekads}")
        t0 = time.time()

        try:
            items = load_dataset(
                short_name,
                bbox=bbox,
                start_date=START,
                end_date=END,
                limit=limit,
                output_dir=output_dir,
                dekads=dekads,
            )
            elapsed = time.time() - t0
            final = load_passes_from_disk(output_dir, dekads=dekads)
            print(f"  -> {len(final)} passes on disk ({elapsed:.0f}s)\n")
            results.append((label, "ok", len(final)))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  -> FAILED after {elapsed:.0f}s: {e}\n")
            traceback.print_exc()
            results.append((label, "failed", str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for label, status, info in results:
        if status == "skipped":
            print(f"  [SKIP] {label}: {info} passes")
        elif status == "ok":
            print(f"  [ OK ] {label}: {info} passes")
        else:
            print(f"  [FAIL] {label}: {info}")

    failed = [r for r in results if r[1] == "failed"]
    if failed:
        print(f"\n{len(failed)} task(s) failed!")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} tasks completed successfully.")


if __name__ == "__main__":
    main()
