"""RS-tools: Remote sensing archive access and visualization toolkit.

Typical end-to-end workflow
---------------------------

.. code-block:: python

    from rs_tools.config import BoundingBox, SearchConfig
    from rs_tools.search import search_archive
    from rs_tools.datasets import (
        summarize_search_results,
        print_coverage_report,
        filter_by_coverage,
        records_to_items,
        load_items,
    )

    # 1. Define search parameters
    bbox = BoundingBox(west=-118.5, south=34.0, east=-117.5, north=35.0)
    config = SearchConfig(
        start_date="2024-01-01",
        end_date="2024-06-30",
        bbox=bbox,
        collections=["OPERA_L2_RTC-S1_V1_1"],
        limit=500,
    )

    # 2. Search returns lightweight metadata — no pixel data downloaded
    items = search_archive("nasa", config)

    # 3. Inspect per-pass coverage over your bbox before committing to a download
    records = summarize_search_results(items, bbox)
    print_coverage_report(records)

    # 4. Filter to passes that cover at least 80 % of the bbox
    selected = filter_by_coverage(records, min_coverage_pct=80.0, orbit_direction="ascending")

    # 5. Load pixel data (lazily via Dask by default, or locally cached)
    loaded = load_items(records_to_items(selected), assets=["VV", "VH"], bbox=bbox, mosaic=True)
"""

__version__ = "0.1.0"
