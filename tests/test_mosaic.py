"""Unit tests for the mosaic module."""

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from rs_tools.datasets.loader import LoadedItem
from rs_tools.datasets.mosaic import group_by_pass, mosaic_items, _snap_to_reference

try:
    import rioxarray  # noqa: F401
    _HAS_RIOXARRAY = True
except ImportError:
    _HAS_RIOXARRAY = False


def _make_simple_item(item_id, dt, orbit="ascending"):
    """Create a minimal LoadedItem with plain xarray data (no rioxarray)."""
    vv = xr.DataArray(np.ones((10, 10), dtype=np.float32), dims=["y", "x"])
    return LoadedItem(
        id=item_id,
        datetime=dt,
        platform="Sentinel-1A",
        orbit_direction=orbit,
        data={"VV": vv},
        crs="EPSG:32631",
        pixel_size_m=30.0,
    )


class TestGroupByPass:
    def test_same_track_same_date(self):
        items = [
            _make_simple_item(
                "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, 51, tzinfo=timezone.utc),
            ),
            _make_simple_item(
                "OPERA_L2_RTC-S1_T059-124884-IW3_20240630T174154Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, 54, tzinfo=timezone.utc),
            ),
        ]
        groups = group_by_pass(items)
        assert len(groups) == 1
        key = (59, "2024-06-30")
        assert key in groups
        assert len(groups[key]) == 2

    def test_different_tracks(self):
        items = [
            _make_simple_item(
                "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, 51, tzinfo=timezone.utc),
            ),
            _make_simple_item(
                "OPERA_L2_RTC-S1_T008-015801-IW1_20240627T060712Z_20240629T074105Z_S1A_30_v1.0",
                datetime(2024, 6, 27, 6, 7, 12, tzinfo=timezone.utc),
            ),
        ]
        groups = group_by_pass(items)
        assert len(groups) == 2

    def test_unparseable_id(self):
        items = [
            _make_simple_item("unknown_item", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ]
        groups = group_by_pass(items)
        assert len(groups) == 0

    def test_same_track_different_dates(self):
        items = [
            _make_simple_item(
                "OPERA_L2_RTC-S1_T161-343970-IW2_20240613T173333Z_20240616T033125Z_S1A_30_v1.0",
                datetime(2024, 6, 13, 17, 33, tzinfo=timezone.utc),
            ),
            _make_simple_item(
                "OPERA_L2_RTC-S1_T161-343970-IW2_20240625T173332Z_20240628T094331Z_S1A_30_v1.0",
                datetime(2024, 6, 25, 17, 33, tzinfo=timezone.utc),
            ),
        ]
        groups = group_by_pass(items)
        assert len(groups) == 2


def _make_rio_item(item_id, dt, orbit="ascending", shape=(10, 10), offset=(0, 0)):
    """Create a LoadedItem with rioxarray-enabled DataArrays."""
    import rioxarray  # noqa: F401

    y_size, x_size = shape
    y_off, x_off = offset
    vv = xr.DataArray(
        np.ones(shape, dtype=np.float32),
        dims=["y", "x"],
        coords={
            "y": np.arange(y_off, y_off + y_size, dtype=np.float64),
            "x": np.arange(x_off, x_off + x_size, dtype=np.float64),
        },
    )
    vv = vv.rio.set_crs("EPSG:32631")
    vv = vv.rio.set_spatial_dims(x_dim="x", y_dim="y")
    vh = vv.copy(deep=True) * 0.5

    return LoadedItem(
        id=item_id,
        datetime=dt,
        platform="Sentinel-1A",
        orbit_direction=orbit,
        data={"VV": vv, "VH": vh},
        crs="EPSG:32631",
        pixel_size_m=30.0,
    )


@pytest.mark.skipif(
    not _HAS_RIOXARRAY,
    reason="rioxarray required for merge tests",
)
class TestMosaicItems:
    def test_single_item_pass_through(self):
        items = [
            _make_rio_item(
                "OPERA_L2_RTC-S1_T037-078095-IW3_20240617T055058Z_20240618T144043Z_S1A_30_v1.0",
                datetime(2024, 6, 17, 5, 50, 58, tzinfo=timezone.utc),
            ),
        ]
        result = mosaic_items(items)
        assert len(result) == 1
        assert result[0] is items[0]

    def test_two_bursts_merged(self):
        items = [
            _make_rio_item(
                "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, 51, tzinfo=timezone.utc),
                shape=(10, 10),
                offset=(0, 0),
            ),
            _make_rio_item(
                "OPERA_L2_RTC-S1_T059-124884-IW3_20240630T174154Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, 54, tzinfo=timezone.utc),
                shape=(10, 10),
                offset=(10, 0),
            ),
        ]
        result = mosaic_items(items)
        assert len(result) == 1
        assert "VV" in result[0].data
        assert "VH" in result[0].data
        assert result[0].platform == "Sentinel-1A"
        assert result[0].orbit_direction == "ascending"
        assert "T059" in result[0].id
        # Merged should be larger than individual
        assert result[0].data["VV"].shape[0] == 20

    def test_mixed_passes(self):
        """Items from two different passes should remain separate."""
        items = [
            _make_rio_item(
                "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
            ),
            _make_rio_item(
                "OPERA_L2_RTC-S1_T059-124884-IW3_20240630T174154Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
                offset=(10, 0),
            ),
            _make_rio_item(
                "OPERA_L2_RTC-S1_T008-015801-IW1_20240627T060712Z_20240629T074105Z_S1A_30_v1.0",
                datetime(2024, 6, 27, 6, 7, tzinfo=timezone.utc),
                orbit="descending",
            ),
        ]
        result = mosaic_items(items)
        assert len(result) == 2
        assert result[0].datetime < result[1].datetime

    def test_result_sorted_by_datetime(self):
        items = [
            _make_rio_item(
                "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
                datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
            ),
            _make_rio_item(
                "OPERA_L2_RTC-S1_T008-015801-IW1_20240627T060712Z_20240629T074105Z_S1A_30_v1.0",
                datetime(2024, 6, 27, 6, 7, tzinfo=timezone.utc),
            ),
            _make_rio_item(
                "OPERA_L2_RTC-S1_T161-343973-IW1_20240625T173340Z_20240628T103258Z_S1A_30_v1.0",
                datetime(2024, 6, 25, 17, 33, tzinfo=timezone.utc),
            ),
        ]
        result = mosaic_items(items)
        assert len(result) == 3
        dates = [r.datetime for r in result]
        assert dates == sorted(dates)


@pytest.mark.skipif(
    not _HAS_RIOXARRAY,
    reason="rioxarray required for snap tests",
)
class TestSnapToReference:
    def test_already_aligned(self):
        """Items with identical grids should pass through unchanged."""
        a = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
            datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
        )
        b = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240712T174151Z_20240712T194030Z_S1A_30_v1.0",
            datetime(2024, 7, 12, 17, 41, tzinfo=timezone.utc),
        )
        result = _snap_to_reference([a, b], a)
        assert result[0].data["VV"].shape == result[1].data["VV"].shape
        np.testing.assert_array_equal(
            result[0].data["VV"].coords["x"].values,
            result[1].data["VV"].coords["x"].values,
        )

    def test_offset_corrected(self):
        """An item with a 0.5-pixel offset should be snapped to reference."""
        ref = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
            datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
            shape=(10, 10),
            offset=(0, 0),
        )
        shifted = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240712T174151Z_20240712T194030Z_S1A_30_v1.0",
            datetime(2024, 7, 12, 17, 41, tzinfo=timezone.utc),
            shape=(10, 10),
            offset=(0.5, 0.5),
        )
        result = _snap_to_reference([ref, shifted], ref)
        # After snap the grids should match the reference
        np.testing.assert_array_equal(
            result[0].data["VV"].coords["x"].values,
            result[1].data["VV"].coords["x"].values,
        )
        np.testing.assert_array_equal(
            result[0].data["VV"].coords["y"].values,
            result[1].data["VV"].coords["y"].values,
        )

    def test_single_item(self):
        """A single-item list should pass through unchanged."""
        a = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
            datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
        )
        result = _snap_to_reference([a], a)
        assert len(result) == 1
        assert result[0] is a

    def test_all_assets_snapped(self):
        """Both VV and VH should be snapped."""
        ref = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
            datetime(2024, 6, 30, 17, 41, tzinfo=timezone.utc),
            shape=(8, 8),
            offset=(0, 0),
        )
        shifted = _make_rio_item(
            "OPERA_L2_RTC-S1_T059-124883-IW3_20240712T174151Z_20240712T194030Z_S1A_30_v1.0",
            datetime(2024, 7, 12, 17, 41, tzinfo=timezone.utc),
            shape=(8, 8),
            offset=(0.3, 0.3),
        )
        result = _snap_to_reference([ref, shifted], ref)
        for asset in ("VV", "VH"):
            np.testing.assert_array_equal(
                result[0].data[asset].coords["x"].values,
                result[1].data[asset].coords["x"].values,
            )
