#!/usr/bin/env python
"""Run the Carbon Engine GPP vs NPP animation headlessly.

Downloads CLMS GPP and NPP 300 m (10-daily / dekadal) over
Western Europe and renders a dual-panel animation GIF.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, items_to_dataarray
from rs_tools.visualization.animation import save_timeseries_gif
from rs_tools.visualization.frames import dual_panel_frames

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/carbon_engine"


def main() -> None:
    bbox = BoundingBox(west=-10, south=35, east=25, north=60)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLMS GPP v2 from CDSE …")
    gpp_items = load_dataset(
        "CLMS_GPP_V2",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/gpp",
    )

    print("Loading CLMS NPP v2 from CDSE …")
    npp_items = load_dataset(
        "CLMS_NPP_V2",
        bbox=bbox,
        start_date="2020-01-01",
        end_date="2023-12-31",
        limit=150,
        output_dir=f"{DATA_DIR}/npp",
    )

    if not gpp_items or not npp_items:
        print("Missing GPP or NPP data — aborting.")
        return

    gpp = items_to_dataarray(gpp_items)
    npp = items_to_dataarray(npp_items)
    print(f"GPP: {gpp.sizes}  NPP: {npp.sizes}")

    print("Rendering dual-panel frames …")
    frames, labels = dual_panel_frames(
        gpp, npp,
        left_cmap="YlGn", right_cmap="YlOrBr",
        left_vmin=0, left_vmax=16,
        right_vmin=-2, right_vmax=10,
        left_label="GPP", right_label="NPP",
        step=1,
    )

    print("Saving GIF …")
    gif_path = save_timeseries_gif(
        frames,
        gif_dir / "carbon_engine_gpp_npp.gif",
        labels=labels,
        title="Carbon Engine — GPP vs NPP",
        fps=4,
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
