#!/usr/bin/env python
"""Run the Amazon vs Sahel dual-panel animation headlessly.

Downloads CLMS NDVI 300 m (10-daily / dekadal) for the Amazon basin
and the Sahel, then renders a side-by-side comparison GIF.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import dual_panel_frames

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/amazon_vs_sahel"


def main() -> None:
    bbox_amazon = BoundingBox(west=-70, south=-10, east=-50, north=5)
    bbox_sahel = BoundingBox(west=-10, south=10, east=15, north=20)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLMS NDVI v3 for Amazon …")
    amazon_items = load_dataset(
        "CLMS_NDVI_V3",
        bbox=bbox_amazon,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/amazon",
    )

    print("Loading CLMS NDVI v3 for Sahel …")
    sahel_items = load_dataset(
        "CLMS_NDVI_V3",
        bbox=bbox_sahel,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/sahel",
    )

    if not amazon_items or not sahel_items:
        print("Missing NDVI data for one or both regions — aborting.")
        return

    ndvi_amazon = items_to_dataarray(amazon_items)
    ndvi_sahel = items_to_dataarray(sahel_items)
    print(f"Amazon: {ndvi_amazon.sizes}  Sahel: {ndvi_sahel.sizes}")

    print("Rendering dual-panel frames …")
    frames, labels = dual_panel_frames(
        ndvi_amazon, ndvi_sahel,
        left_cmap="YlGn", right_cmap="YlGn",
        left_vmin=0, left_vmax=0.9,
        right_vmin=0, right_vmax=0.9,
        left_label="Amazon", right_label="Sahel",
        step=1,
    )

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        frames,
        gif_dir / "amazon_vs_sahel_ndvi.gif",
        labels=labels,
        title="Amazon vs Sahel — NDVI",
        fps=4,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
