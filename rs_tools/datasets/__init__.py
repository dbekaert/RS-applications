"""Dataset catalog, loading, and coverage utilities."""

from rs_tools.datasets.backscatter import (
    BackscatterType,
    apply_anf,
    convert_pass_backscatter,
    extract_burst_id,
)
from rs_tools.datasets.catalog import (
    DatasetInfo,
    get,
    list_datasets,
    register,
)
from rs_tools.datasets.coverage import (
    PassRecord,
    compute_bbox_coverage,
    filter_by_coverage,
    print_coverage_report,
    records_to_items,
    summarize_search_results,
)
from rs_tools.datasets.loader import (
    LoadedItem,
    deduplicate_items,
    extract_item_metadata,
    load_dataset,
    load_items,
    load_passes_from_disk,
    parse_opera_rtc_id,
    subsample_monthly,
)

__all__ = [
    # backscatter
    "BackscatterType",
    "apply_anf",
    "convert_pass_backscatter",
    "extract_burst_id",
    # catalog
    "DatasetInfo",
    "get",
    "list_datasets",
    "register",
    # coverage
    "PassRecord",
    "compute_bbox_coverage",
    "filter_by_coverage",
    "print_coverage_report",
    "records_to_items",
    "summarize_search_results",
    # loader
    "LoadedItem",
    "deduplicate_items",
    "extract_item_metadata",
    "load_dataset",
    "load_items",
    "parse_opera_rtc_id",
    "subsample_monthly",
]
