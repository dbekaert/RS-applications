"""Unit tests for visualization helpers (non-interactive)."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from rs_tools.visualization.rgb_composite import make_rgb, multi_temporal_rgb
from rs_tools.visualization.rtc_composite import rtc_composite
from rs_tools.visualization.scalebar import add_scalebar, _auto_length_km
from rs_tools.visualization.animation import save_timeseries_gif


class TestMakeRGB:
    def test_output_shape(self):
        r = np.random.rand(100, 100)
        g = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        rgb = make_rgb(r, g, b)
        assert rgb.shape == (100, 100, 3)

    def test_values_in_range(self):
        arr = np.random.rand(50, 50)
        rgb = make_rgb(arr, arr, arr)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


class TestMultiTemporalRGB:
    def test_basic(self):
        data = xr.DataArray(
            np.random.rand(5, 50, 50),
            dims=["time", "y", "x"],
        )
        rgb = multi_temporal_rgb(data, time_indices=(0, 2, 4))
        assert rgb.shape == (50, 50, 3)


class TestRTCComposite:
    def test_output_shape(self):
        vv = np.random.rand(100, 100) * 0.3
        vh = np.random.rand(100, 100) * 0.1
        rgb = rtc_composite(vv, vh)
        assert rgb.shape == (100, 100, 3)

    def test_values_in_range(self):
        vv = np.random.rand(50, 50) * 0.3
        vh = np.random.rand(50, 50) * 0.1
        rgb = rtc_composite(vv, vh)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_nan_handling(self):
        vv = np.full((10, 10), 0.2)
        vh = np.full((10, 10), 0.05)
        vv[5, 5] = np.nan
        vh[5, 5] = np.nan
        rgb = rtc_composite(vv, vh)
        assert rgb[5, 5, 0] == 0.0
        assert rgb[5, 5, 1] == 0.0

    def test_r_equals_b(self):
        vv = np.random.rand(20, 20) * 0.3
        vh = np.random.rand(20, 20) * 0.1
        rgb = rtc_composite(vv, vh)
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 2])


class TestScalebar:
    def test_auto_length(self):
        length = _auto_length_km(1000, 30.0)
        assert length in [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

    def test_add_scalebar(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(100, 100))
        sb = add_scalebar(ax, pixel_size_m=30.0, length_km=1.0)
        assert sb is not None
        plt.close(fig)

    def test_auto_scalebar(self):
        fig, ax = plt.subplots()
        ax.imshow(np.random.rand(200, 200))
        sb = add_scalebar(ax, pixel_size_m=30.0)
        assert sb is not None
        plt.close(fig)


class TestAnimation:
    def test_save_gif(self):
        frames = [np.random.rand(50, 50, 3) for _ in range(3)]
        labels = ["Frame 1", "Frame 2", "Frame 3"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_timeseries_gif(
                frames,
                os.path.join(tmpdir, "test.gif"),
                labels=labels,
                title="Test",
                fps=2,
                figsize=(4, 4),
                dpi=50,
            )
            assert path.exists()
            assert path.stat().st_size > 0

    def test_save_gif_with_scalebar(self):
        frames = [np.random.rand(50, 50, 3) for _ in range(2)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_timeseries_gif(
                frames,
                os.path.join(tmpdir, "test_sb.gif"),
                pixel_size_m=30.0,
                scalebar_km=1.0,
                fps=1,
                figsize=(4, 4),
                dpi=50,
            )
            assert path.exists()
