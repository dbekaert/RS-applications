#!/usr/bin/env python
"""Run the Drought Pulse SWI vs ETA animation headlessly.

Downloads CLMS SWI (daily, 12.5 km) and ETA (10-daily / dekadal,
300 m) over Central Europe and renders a dual-panel animation
highlighting the 2022 drought.

Note: SWI is a daily product at 12.5 km resolution while ETA is
a dekadal product at 300 m — different spatial and temporal grids.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import dual_panel_frames

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/drought_pulse"


def main() -> None:
    bbox = BoundingBox(west=-5, south=42, east=20, north=55)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLMS SWI v4 from CDSE …")
    swi_items = load_dataset(
        "CLMS_SWI_V4",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/swi",
    )

    print("Loading CLMS ETA v1 from CDSE …")
    eta_items = load_dataset(
        "CLMS_ETA_V1",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/eta",
    )

    if not swi_items or not eta_items:
        print("Missing SWI or ETA data — aborting.")
        return

    swi = items_to_dataarray(swi_items)
    eta = items_to_dataarray(eta_items)
    print(f"SWI: {swi.sizes}  ETA: {eta.sizes}")

    print("Rendering dual-panel frames …")
    frames, labels = dual_panel_frames(
        swi, eta,
        left_cmap="YlGnBu", right_cmap="YlOrRd",
        left_vmin=10, left_vmax=90,
        right_vmin=0, right_vmax=5,
        left_label="SWI", right_label="ETA",
        step=1,
    )

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        frames,
        gif_dir / "drought_pulse_swi_eta.gif",
        labels=labels,
        title="Drought Pulse — SWI vs ETA",
        fps=4,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
