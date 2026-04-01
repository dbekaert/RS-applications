#!/usr/bin/env python
"""Run the Fire & Recovery overlay animation headlessly.

Downloads CLMS NDVI (10-daily / dekadal) and Burnt Area (monthly)
over the Iberian Peninsula and renders an overlay animation GIF.

Usage:
    python run_fire_recovery.py                  # all dekads
    python run_fire_recovery.py --dekads 1       # dekad 1 only
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from functools import partial
from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.frames import make_overlay_composite

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/fire_recovery"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire & Recovery overlay animation",
    )
    parser.add_argument(
        "--dekads", nargs="+", type=int, choices=[1, 2, 3], default=None,
        help="Dekad(s) to include (1, 2, 3). Default: all.",
    )
    args = parser.parse_args()

    bbox = BoundingBox(west=-10, south=36, east=4, north=44)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    dekads = args.dekads

    print("Loading CLMS NDVI v3 from CDSE …")
    load_dataset(
        "CLMS_NDVI_V3", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=f"{DATA_DIR}/ndvi", dekads=dekads,
    )

    print("Loading CLMS Burnt Area v4 (monthly) from CDSE …")
    load_dataset(
        "CLMS_BA_V4_MONTHLY", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=50, output_dir=f"{DATA_DIR}/ba", dekads=dekads,
    )

    ndvi_items = load_passes_from_disk(f"{DATA_DIR}/ndvi", dekads=dekads)
    ba_items = load_passes_from_disk(f"{DATA_DIR}/ba", dekads=dekads)
    if not ndvi_items or not ba_items:
        print("Missing NDVI or BA data — aborting.")
        return

    n = min(len(ndvi_items), len(ba_items))
    ndvi_items, ba_items = ndvi_items[:n], ba_items[:n]
    print(f"NDVI: {n} dekads  BA: {n} items")

    overlay_composite = partial(
        make_overlay_composite,
        base_cmap="YlGn", overlay_cmap="Reds",
        base_vmin=0, base_vmax=0.9,
        overlay_threshold=0.1, overlay_alpha=0.7,
    )

    def _composite(pair):
        base, overlay = pair
        return overlay_composite(base, overlay)

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"fire_recovery_iberia_D{suffix}.gif"

    print("Saving GIF (lazy — one frame at a time) …")
    gif_path = save_timeseries_gif_lazy(
        zip(ndvi_items, ba_items),
        gif_dir / gif_name,
        composite_fn=_composite,
        title="Fire & Recovery — Iberian Peninsula",
        fps=6,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
