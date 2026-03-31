#!/usr/bin/env python
"""Run the Breathing Pulse NDVI animation headlessly.

Downloads CLMS NDVI 300 m (10-daily / dekadal) over Western Europe
and renders a seasonal animation GIF.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import data_to_rgb_frames

# Working directory for downloaded data
DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/breathing_pulse"


def main() -> None:
    bbox = BoundingBox(west=-10, south=35, east=25, north=60)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLMS NDVI v3 from CDSE …")
    items = load_dataset(
        "CLMS_NDVI_V3",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=DATA_DIR,
    )
    if not items:
        print("No NDVI items found — aborting.")
        return

    ndvi = items_to_dataarray(items)
    print(f"NDVI time-series: {ndvi.sizes}")

    print("Rendering frames …")
    frames, labels = data_to_rgb_frames(ndvi, cmap="YlGn", vmin=0, vmax=0.9, step=1)

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        frames,
        gif_dir / "breathing_pulse_ndvi.gif",
        labels=labels,
        title="NDVI — Earth's Breathing Pulse",
        fps=4,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
