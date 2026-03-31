"""Unit tests for rs_tools.visualization.frames module."""

import numpy as np
import pytest
import xarray as xr

from rs_tools.config import BoundingBox
from rs_tools.datasets.synthetic import (
    generate_burnt_area,
    generate_fapar,
    generate_gpp,
    generate_lai,
    generate_ndvi,
    generate_ndvi_with_fire,
    generate_npp,
)
from rs_tools.visualization.frames import (
    data_to_rgb_frames,
    dual_panel_frames,
    overlay_frames,
    product_cycle_frames,
)


@pytest.fixture
def bbox():
    return BoundingBox(west=0, south=40, east=10, north=50)


@pytest.fixture
def ndvi(bbox):
    return generate_ndvi(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)


@pytest.fixture
def gpp(bbox):
    return generate_gpp(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)


@pytest.fixture
def npp(bbox):
    return generate_npp(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)


# -----------------------------------------------------------------------
# data_to_rgb_frames
# -----------------------------------------------------------------------

class TestDataToRgbFrames:

    def test_output_shape(self, ndvi):
        frames, labels = data_to_rgb_frames(ndvi, cmap="YlGn")
        assert len(frames) == len(ndvi.time)
        assert len(labels) == len(ndvi.time)
        assert frames[0].shape == (20, 30, 3)

    def test_values_in_01(self, ndvi):
        frames, _ = data_to_rgb_frames(ndvi, cmap="YlGn")
        for f in frames:
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_step_subsampling(self, ndvi):
        frames_all, _ = data_to_rgb_frames(ndvi, step=1)
        frames_sub, _ = data_to_rgb_frames(ndvi, step=3)
        assert len(frames_sub) < len(frames_all)
        expected = len(range(0, len(ndvi.time), 3))
        assert len(frames_sub) == expected

    def test_labels_are_dates(self, ndvi):
        _, labels = data_to_rgb_frames(ndvi)
        assert "2021" in labels[0]

    def test_explicit_vmin_vmax(self, ndvi):
        frames, _ = data_to_rgb_frames(ndvi, vmin=0, vmax=1)
        assert len(frames) > 0


# -----------------------------------------------------------------------
# dual_panel_frames
# -----------------------------------------------------------------------

class TestDualPanelFrames:

    def test_output_shape(self, gpp, npp):
        frames, labels = dual_panel_frames(gpp, npp, step=3)
        assert len(frames) > 0
        assert len(labels) == len(frames)
        # Width = left + gap + right
        expected_w = gpp.shape[2] + 4 + npp.shape[2]
        assert frames[0].shape == (20, expected_w, 3)

    def test_labels_contain_product_names(self, gpp, npp):
        frames, labels = dual_panel_frames(
            gpp, npp, left_label="GPP", right_label="NPP", step=6,
        )
        assert "GPP" in labels[0]
        assert "NPP" in labels[0]

    def test_custom_gap(self, gpp, npp):
        frames, _ = dual_panel_frames(gpp, npp, gap_px=10, step=6)
        expected_w = gpp.shape[2] + 10 + npp.shape[2]
        assert frames[0].shape[1] == expected_w

    def test_values_in_01(self, gpp, npp):
        frames, _ = dual_panel_frames(gpp, npp, step=6)
        for f in frames:
            assert f.min() >= 0.0
            assert f.max() <= 1.0


# -----------------------------------------------------------------------
# overlay_frames
# -----------------------------------------------------------------------

class TestOverlayFrames:

    def test_basic_overlay(self, bbox):
        ndvi = generate_ndvi_with_fire(
            bbox, "2021-01-01", "2021-12-31", ny=20, nx=30,
            fire_times=[(5, 10, 15, 5)],
        )
        ba = generate_burnt_area(
            bbox, "2021-01-01", "2021-12-31", ny=20, nx=30,
            fire_times=[(5, 10, 15, 5)],
        )
        frames, labels = overlay_frames(ndvi, ba, step=3)
        assert len(frames) > 0
        assert frames[0].shape == (20, 30, 3)

    def test_no_overlay_passthrough(self, bbox):
        ndvi = generate_ndvi(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        ba = generate_burnt_area(
            bbox, "2021-01-01", "2021-06-30", ny=20, nx=30, fire_times=[],
        )
        frames, _ = overlay_frames(ndvi, ba, step=6)
        base_frames, _ = data_to_rgb_frames(ndvi, cmap="YlGn", step=6)
        # With no fire, overlay frames should equal base frames
        np.testing.assert_array_almost_equal(frames[0], base_frames[0], decimal=5)


# -----------------------------------------------------------------------
# product_cycle_frames
# -----------------------------------------------------------------------

class TestProductCycleFrames:

    def test_one_frame_per_product(self, bbox):
        ndvi = generate_ndvi(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        lai = generate_lai(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        products = {"NDVI": ndvi, "LAI": lai}
        frames, labels = product_cycle_frames(products, time_index=0)
        assert len(frames) == 2
        assert labels == ["NDVI", "LAI"]

    def test_frame_shapes(self, bbox):
        ndvi = generate_ndvi(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        fapar = generate_fapar(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        products = {"NDVI": ndvi, "FAPAR": fapar}
        frames, _ = product_cycle_frames(products, time_index=0)
        assert frames[0].shape == (20, 30, 3)
        assert frames[1].shape == (20, 30, 3)

    def test_custom_cmaps(self, bbox):
        ndvi = generate_ndvi(bbox, "2021-01-01", "2021-06-30", ny=20, nx=30)
        products = {"NDVI": ndvi}
        cmaps = {"NDVI": "Greens"}
        frames, _ = product_cycle_frames(products, cmaps=cmaps, time_index=0)
        assert len(frames) == 1
