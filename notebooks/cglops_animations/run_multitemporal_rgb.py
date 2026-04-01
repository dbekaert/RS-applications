#!/usr/bin/env python
"""Run the Multi-Temporal RGB Seasons animation headlessly.

Downloads CLMS NDVI 300 m (10-daily / dekadal) over Western Europe
and builds per-year R=winter / G=spring / B=summer composites.

Note: This workflow requires random time-index access across the
full series to pick 3 specific dates per year, so
``items_to_dataarray`` is still used.  Memory pressure is low (only
4 output frames).

Usage:
    python run_multitemporal_rgb.py                  # all dekads
    python run_multitemporal_rgb.py --dekads 1       # dekad 1 only
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk, items_to_dataarray
from rs_tools.visualization.rgb_composite import multi_temporal_rgb
from rs_tools.visualization.animation import save_timeseries_gif

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/multitemporal_rgb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Temporal RGB Seasons animation",
    )
    parser.add_argument(
        "--dekads", nargs="+", type=int, choices=[1, 2, 3], default=None,
        help="Dekad(s) to include (1, 2, 3). Default: all.",
    )
    args = parser.parse_args()

    bbox = BoundingBox(west=-10, south=35, east=25, north=60)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    dekads = args.dekads

    print("Loading CLMS NDVI v3 from CDSE …")
    load_dataset(
        "CLMS_NDVI_V3", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=DATA_DIR, dekads=dekads,
    )

    items = load_passes_from_disk(DATA_DIR, dekads=dekads)
    if not items:
        print("No NDVI items found — aborting.")
        return

    ndvi = items_to_dataarray(items)
    print(f"NDVI time-series: {ndvi.sizes}")

    years = sorted({int(str(t)[:4]) for t in ndvi.time.values})
    rgb_composites = []
    year_labels = []

    for yr in years:
        times = ndvi.time.values
        targets = [
            np.datetime64(f"{yr}-01-15"),
            np.datetime64(f"{yr}-04-15"),
            np.datetime64(f"{yr}-07-15"),
        ]
        indices = tuple(int(np.argmin(np.abs(times - t))) for t in targets)
        rgb = multi_temporal_rgb(ndvi, indices, vmin=0.1, vmax=0.85)
        rgb_composites.append(rgb)
        year_labels.append(str(yr))

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"multitemporal_rgb_seasons_D{suffix}.gif"

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        rgb_composites,
        gif_dir / gif_name,
        labels=year_labels,
        title="Seasonal NDVI RGB — R:Winter G:Spring B:Summer",
        fps=1,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
