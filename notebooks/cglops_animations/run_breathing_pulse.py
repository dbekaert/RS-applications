#!/usr/bin/env python
"""Run the Breathing Pulse NDVI animation headlessly.

Downloads CLMS NDVI 300 m (10-daily / dekadal) over Western Europe
and renders a seasonal animation GIF.

Usage:
    python run_breathing_pulse.py                     # all dekads
    python run_breathing_pulse.py --dekads 1          # dekad 1 only
    python run_breathing_pulse.py --dekads 1 2        # dekads 1 & 2
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from functools import partial
from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.frames import make_colormap_composite

# Working directory for downloaded data
DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/breathing_pulse"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Breathing Pulse NDVI animation",
    )
    parser.add_argument(
        "--dekads",
        nargs="+",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Dekad(s) to include: 1 (day 1-10), 2 (day 11-20), "
             "3 (day 21-end). Default: all.",
    )
    args = parser.parse_args()

    bbox = BoundingBox(west=-10, south=35, east=25, north=60)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    dekads = args.dekads

    print("Loading CLMS NDVI v3 from CDSE …")
    items = load_dataset(
        "CLMS_NDVI_V3",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2026-03-01",
        limit=150,
        output_dir=DATA_DIR,
        dekads=dekads,
    )
    if not items:
        print("No NDVI items found — aborting.")
        return

    # Reload as lightweight references (pixel data stays on disk)
    items = load_passes_from_disk(DATA_DIR, dekads=dekads)
    print(f"{len(items)} NDVI dekads available on disk")

    # Build GIF filename reflecting dekad selection
    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"breathing_pulse_ndvi_D{suffix}.gif"

    print("Saving GIF (lazy — one frame at a time) …")
    ndvi_composite = partial(make_colormap_composite, cmap="YlGn", vmin=0, vmax=0.9)

    gif_path = save_timeseries_gif_lazy(
        items,
        gif_dir / gif_name,
        composite_fn=ndvi_composite,
        title="NDVI — Earth's Breathing Pulse",
        fps=4,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
