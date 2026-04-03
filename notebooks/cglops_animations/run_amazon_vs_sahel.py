#!/usr/bin/env python
"""Run the Amazon vs Sahel dual-panel animation headlessly.

Downloads CLMS NDVI 300 m (10-daily / dekadal) for the Amazon basin
and the Sahel, then renders a side-by-side comparison GIF.

Usage:
    python run_amazon_vs_sahel.py                  # all dekads
    python run_amazon_vs_sahel.py --dekads 1       # dekad 1 only
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from functools import partial
from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk
from rs_tools.visualization.animation import save_timeseries_gif_lazy
from rs_tools.visualization.frames import make_dual_panel_composite
from rs_tools.visualization.clms_colormaps import CLMS_NDVI, NDVI_VMIN, NDVI_VMAX

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/amazon_vs_sahel"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Amazon vs Sahel NDVI animation",
    )
    parser.add_argument(
        "--dekads", nargs="+", type=int, choices=[1, 2, 3], default=None,
        help="Dekad(s) to include (1, 2, 3). Default: all.",
    )
    args = parser.parse_args()

    bbox_amazon = BoundingBox(west=-70, south=-10, east=-50, north=5)
    bbox_sahel = BoundingBox(west=-10, south=10, east=15, north=20)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    dekads = args.dekads

    print("Loading CLMS NDVI v3 for Amazon …")
    load_dataset(
        "CLMS_NDVI_V3", bbox=bbox_amazon,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=500, output_dir=f"{DATA_DIR}/amazon", dekads=dekads,
    )

    print("Loading CLMS NDVI v3 for Sahel …")
    load_dataset(
        "CLMS_NDVI_V3", bbox=bbox_sahel,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=500, output_dir=f"{DATA_DIR}/sahel", dekads=dekads,
    )

    amazon_items = load_passes_from_disk(f"{DATA_DIR}/amazon", dekads=dekads)
    sahel_items = load_passes_from_disk(f"{DATA_DIR}/sahel", dekads=dekads)
    if not amazon_items or not sahel_items:
        print("Missing NDVI data for one or both regions — aborting.")
        return

    n = min(len(amazon_items), len(sahel_items))
    amazon_items, sahel_items = amazon_items[:n], sahel_items[:n]
    print(f"Amazon: {n} dekads  Sahel: {n} dekads")

    dual_composite = partial(
        make_dual_panel_composite,
        left_cmap=CLMS_NDVI, right_cmap=CLMS_NDVI,
        left_vmin=NDVI_VMIN, left_vmax=NDVI_VMAX,
        right_vmin=NDVI_VMIN, right_vmax=NDVI_VMAX,
        left_label="Amazon", right_label="Sahel",
    )

    def _composite(pair):
        left, right = pair
        return dual_composite(left, right)

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"amazon_vs_sahel_ndvi_D{suffix}.gif"

    print("Saving GIF (lazy — one frame at a time) …")
    gif_path = save_timeseries_gif_lazy(
        zip(amazon_items, sahel_items),
        gif_dir / gif_name,
        composite_fn=_composite,
        title="Amazon vs Sahel — NDVI",
        fps=4,
        figsize=(16, 8),
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
