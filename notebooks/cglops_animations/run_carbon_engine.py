#!/usr/bin/env python
"""Run the Carbon Engine GPP vs NPP animation headlessly.

Downloads CLMS GPP and NPP 300 m (10-daily / dekadal) over
Western Europe and renders a dual-panel animation GIF.

Usage:
    python run_carbon_engine.py                  # all dekads
    python run_carbon_engine.py --dekads 1       # dekad 1 only
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
from rs_tools.visualization.clms_colormaps import CLMS_GPP, GPP_VMIN, GPP_VMAX, CLMS_NPP, NPP_VMIN, NPP_VMAX

DATA_DIR = "/home/bekaertd/RS_applications/Applications/CGOPS/carbon_engine"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carbon Engine GPP vs NPP animation",
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

    print("Loading CLMS GPP v2 from CDSE …")
    load_dataset(
        "CLMS_GPP_V2", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=f"{DATA_DIR}/gpp", dekads=dekads,
    )

    print("Loading CLMS NPP v2 from CDSE …")
    load_dataset(
        "CLMS_NPP_V2", bbox=bbox,
        start_date="2020-01-01", end_date="2026-03-01",
        limit=150, output_dir=f"{DATA_DIR}/npp", dekads=dekads,
    )

    gpp_items = load_passes_from_disk(f"{DATA_DIR}/gpp", dekads=dekads)
    npp_items = load_passes_from_disk(f"{DATA_DIR}/npp", dekads=dekads)
    if not gpp_items or not npp_items:
        print("Missing GPP or NPP data — aborting.")
        return

    n = min(len(gpp_items), len(npp_items))
    gpp_items, npp_items = gpp_items[:n], npp_items[:n]
    print(f"GPP: {n} dekads  NPP: {n} dekads")

    dual_composite = partial(
        make_dual_panel_composite,
        left_cmap=CLMS_GPP, right_cmap=CLMS_NPP,
        left_vmin=GPP_VMIN, left_vmax=GPP_VMAX,
        right_vmin=NPP_VMIN, right_vmax=NPP_VMAX,
        left_label="GPP", right_label="NPP",
    )

    def _composite(pair):
        left, right = pair
        return dual_composite(left, right)

    suffix = "_".join(str(d) for d in dekads) if dekads else "all"
    gif_name = f"carbon_engine_gpp_npp_D{suffix}.gif"

    print("Saving GIF (lazy — one frame at a time) …")
    gif_path = save_timeseries_gif_lazy(
        zip(gpp_items, npp_items),
        gif_dir / gif_name,
        composite_fn=_composite,
        title="Carbon Engine — GPP vs NPP",
        fps=4,
        figsize=(16, 8),
    )
    print(f"Done: {gif_path}")


if __name__ == "__main__":
    main()
