#!/usr/bin/env python
"""Run the Vegetation Anatomy product-tour animation headlessly.

Downloads CLMS NDVI, LAI, FAPAR, and FCOVER 300 m (10-daily /
dekadal) over Western Europe and renders a rotating product-tour GIF.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import product_cycle_frames

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/vegetation_anatomy"


def main() -> None:
    bbox = BoundingBox(west=-10, south=35, east=25, north=60)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    products = {
        "NDVI": "CLMS_NDVI_V3",
        "LAI": "CLMS_LAI_V2",
        "FAPAR": "CLMS_FAPAR_V2",
        "FCOVER": "CLMS_FCOVER_V2",
    }

    product_dict = {}
    for label, short_name in products.items():
        print(f"Loading {short_name} from CDSE …")
        items = load_dataset(
            short_name,
            bbox=bbox,
            start_date="2020-01-01",
            end_date="2023-12-31",
            limit=150,
            output_dir=f"{DATA_DIR}/{label.lower()}",
        )
        if not items:
            print(f"No {label} items found — aborting.")
            return
        product_dict[label] = items_to_dataarray(items)
        print(f"  {label}: {product_dict[label].sizes}")

    cmaps = {"NDVI": "YlGn", "LAI": "Greens", "FAPAR": "YlGn", "FCOVER": "BuGn"}
    vmins = {"NDVI": 0, "LAI": 0, "FAPAR": 0, "FCOVER": 0}
    vmaxs = {"NDVI": 0.9, "LAI": 6, "FAPAR": 0.8, "FCOVER": 0.7}

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

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        all_frames,
        gif_dir / "vegetation_anatomy_tour.gif",
        labels=all_labels,
        title="Vegetation Anatomy — Product Tour",
        fps=2,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
