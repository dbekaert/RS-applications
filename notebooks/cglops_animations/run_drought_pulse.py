#!/usr/bin/env python
"""Run the Drought Pulse SWI vs ETA animation headlessly.

Downloads CLMS SWI (daily, 12.5 km) and ETA (10-daily / dekadal,
300 m) over Central Europe and renders a dual-panel animation
highlighting the 2022 drought.

Usage:
    python run_drought_pulse.py                  # all dekads
    python run_drought_pulse.py --dekads 1       # dekad 1 only
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

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/drought_pulse"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drought Pulse SWI vs ETA animation",
    )
    parser.add_argument(
        "--dekads", nargs="+", type=int, choices=[1, 2, 3], default=None,
        help="Dekad(s) to include (1, 2, 3). Default: all.",
    )
    args = parser.parse_args()

    bbox = BoundingBox(west=-5, south=42, east=20, north=55)
    gif_dir = Path(__file__).resolve().parent / "output" / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    dekads = args.dekads

    print("Loading CLMS SWI v4 from CDSE …")
    load_dataset(
        "CLMS_SWI_V4", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=f"{DATA_DIR}/swi", dekads=dekads,
    )

    print("Loading CLMS ETA v1 from CDSE …")
    load_dataset(
        "CLMS_ETA_V1", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=f"{DATA_DIR}/eta", dekads=dekads,
    )

    swi_items = load_passes_from_disk(f"{DATA_DIR}/swi", dekads=dekads)
    eta_items = load_passes_from_disk(f"{DATA_DIR}/eta", dekads=dekads)
    if not swi_items or not eta_items:
        print("Missing SWI or ETA data — aborting.")
        return

    n = min(len(swi_items), len(eta_items))
    swi_items, eta_items = swi_items[:n], eta_items[:n]
    print(f"SWI: {n} items  ETA: {n} dekads")

    dual_composite = partial(
        make_dual_panel_composite,
        left_cmap="YlGnBu", right_cmap="YlOrRd",
        left_vmin=10, left_vmax=90,
        right_vmin=0, right_vmax=5,
        left_label="SWI", right_label="ETA",
    )

    def _composite(pair):
        left, right = pair
        return dual_composite(left, right)

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"drought_pulse_swi_eta_D{suffix}.gif"

    print("Saving GIF (lazy — one frame at a time) …")
    gif_path = save_timeseries_gif_lazy(
        zip(swi_items, eta_items),
        gif_dir / gif_name,
        composite_fn=_composite,
        title="Drought Pulse — SWI vs ETA",
        fps=4,
        figsize=(16, 8),
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
