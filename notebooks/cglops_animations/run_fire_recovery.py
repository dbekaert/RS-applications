#!/usr/bin/env python
"""Run the Fire & Recovery overlay animation headlessly.

Downloads CLMS NDVI (10-daily / dekadal) and Burnt Area (monthly)
over the Iberian Peninsula and renders an overlay animation GIF.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import overlay_frames

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/fire_recovery"


def main() -> None:
    bbox = BoundingBox(west=-10, south=36, east=4, north=44)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLMS NDVI v3 from CDSE …")
    ndvi_items = load_dataset(
        "CLMS_NDVI_V3",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/ndvi",
    )

    print("Loading CLMS Burnt Area v4 (monthly) from CDSE …")
    ba_items = load_dataset(
        "CLMS_BA_V4_MONTHLY",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=50,
        output_dir=f"{DATA_DIR}/ba",
    )

    if not ndvi_items or not ba_items:
        print("Missing NDVI or BA data — aborting.")
        return

    ndvi = items_to_dataarray(ndvi_items)
    ba = items_to_dataarray(ba_items)
    print(f"NDVI: {ndvi.sizes}  BA: {ba.sizes}")

    print("Rendering overlay frames …")
    frames, labels = overlay_frames(
        ndvi, ba,
        base_cmap="YlGn", overlay_cmap="Reds",
        base_vmin=0, base_vmax=0.9,
        overlay_threshold=0.1,
        overlay_alpha=0.7,
        step=1,
    )

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        frames,
        gif_dir / "fire_recovery_iberia.gif",
        labels=labels,
        title="Fire & Recovery — Iberian Peninsula",
        fps=6,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
