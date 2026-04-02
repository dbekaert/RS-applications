#!/usr/bin/env python
"""Run the Vegetation Anatomy product-tour animation headlessly.

Downloads CLMS NDVI, LAI, FAPAR, and FCOVER 300 m (10-daily /
dekadal) over Western Europe and renders a rotating product-tour GIF.

Note: This workflow requires random time-index access across the
full series for the product-cycle layout, so ``items_to_dataarray``
is still used.

Usage:
    python run_vegetation_anatomy.py                  # all dekads
    python run_vegetation_anatomy.py --dekads 1       # dekad 1 only
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, load_passes_from_disk, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import product_cycle_frames
from rs_tools.visualization.clms_colormaps import CLMS_COLORMAPS, CLMS_VMINS, CLMS_VMAXS

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/vegetation_anatomy"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vegetation Anatomy product-tour animation",
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

    products = {
        "NDVI": "CLMS_NDVI_V3",
        "LAI": "CLMS_LAI_V2",
        "FAPAR": "CLMS_FAPAR_V2",
        "FCOVER": "CLMS_FCOVER_V2",
    }

    product_dict = {}
    for label, short_name in products.items():
        print(f"Loading {short_name} from CDSE …")
        load_dataset(
            short_name, bbox=bbox,
            start_date="2020-01-01", end_date="2026-03-01",
            limit=150, output_dir=f"{DATA_DIR}/{label.lower()}",
            dekads=dekads,
        )
        items = load_passes_from_disk(
            f"{DATA_DIR}/{label.lower()}", dekads=dekads,
        )
        if not items:
            print(f"No {label} items found — aborting.")
            return
        product_dict[label] = items_to_dataarray(items)
        print(f"  {label}: {product_dict[label].sizes}")

    cmaps = {k: CLMS_COLORMAPS[k] for k in products}
    vmins = {k: CLMS_VMINS[k] for k in products}
    vmaxs = {k: CLMS_VMAXS[k] for k in products}

    ref_da = product_dict["NDVI"]
    all_frames = []
    all_labels = []
    for t in range(0, len(ref_da.time), 3):
        frames, labels = product_cycle_frames(
            product_dict, cmaps=cmaps, vmins=vmins, vmaxs=vmaxs, time_index=t,
        )
        date_str = str(np.datetime_as_string(ref_da.time.values[t], unit="D"))
        for f, lbl in zip(frames, labels):
            all_frames.append(f)
            all_labels.append(f"{date_str}  —  {lbl}")

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"vegetation_anatomy_tour_D{suffix}.gif"

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        all_frames,
        gif_dir / gif_name,
        labels=all_labels,
        title="Vegetation Anatomy — Product Tour",
        fps=2,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
