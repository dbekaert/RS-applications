"""Unit tests for rs_tools.datasets.synthetic generators."""

import numpy as np
import pytest
import xarray as xr

from rs_tools.config import BoundingBox
from rs_tools.datasets.synthetic import (
    generate_burnt_area,
    generate_eta,
    generate_fapar,
    generate_fcover,
    generate_gpp,
    generate_lai,
    generate_ndvi,
    generate_ndvi_with_fire,
    generate_npp,
    generate_swi,
)


@pytest.fixture
def bbox():
    return BoundingBox(west=-10, south=35, east=25, north=60)


@pytest.fixture
def small_kwargs():
    """Small grid for fast tests."""
    return dict(start="2021-01-01", end="2021-12-31", ny=20, nx=30)


# -----------------------------------------------------------------------
# Shape & coordinate tests
# -----------------------------------------------------------------------

class TestGeneratorShapes:
    """All generators should produce correctly shaped DataArrays."""

    @pytest.mark.parametrize("gen_fn", [
        generate_ndvi, generate_lai, generate_fapar, generate_fcover,
        generate_gpp, generate_npp, generate_eta,
    ])
    def test_shape_and_dims(self, bbox, small_kwargs, gen_fn):
        da = gen_fn(bbox, **small_kwargs)
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time", "y", "x")
        assert da.shape[1] == small_kwargs["ny"]
        assert da.shape[2] == small_kwargs["nx"]
        assert da.shape[0] > 0

    def test_swi_shape(self, bbox, small_kwargs):
        da = generate_swi(bbox, **small_kwargs, drought_year=None)
        assert da.dims == ("time", "y", "x")
        assert da.shape[1] == small_kwargs["ny"]

    def test_burnt_area_shape(self, bbox, small_kwargs):
        da = generate_burnt_area(bbox, **small_kwargs)
        assert da.dims == ("time", "y", "x")

    def test_ndvi_with_fire_shape(self, bbox, small_kwargs):
        da = generate_ndvi_with_fire(bbox, **small_kwargs)
        assert da.dims == ("time", "y", "x")


# -----------------------------------------------------------------------
# Value range tests
# -----------------------------------------------------------------------

class TestValueRanges:
    """Each product should stay within its physical range."""

    def test_ndvi_range(self, bbox, small_kwargs):
        da = generate_ndvi(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 1.0

    def test_lai_range(self, bbox, small_kwargs):
        da = generate_lai(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 7.0

    def test_fapar_range(self, bbox, small_kwargs):
        da = generate_fapar(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 1.0

    def test_fcover_range(self, bbox, small_kwargs):
        da = generate_fcover(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 1.0

    def test_gpp_range(self, bbox, small_kwargs):
        da = generate_gpp(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 20.0

    def test_npp_range(self, bbox, small_kwargs):
        da = generate_npp(bbox, **small_kwargs)
        assert float(da.min()) >= -2.0
        assert float(da.max()) <= 12.0

    def test_eta_range(self, bbox, small_kwargs):
        da = generate_eta(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 6.0

    def test_swi_range(self, bbox, small_kwargs):
        da = generate_swi(bbox, **small_kwargs, drought_year=None)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 100.0

    def test_burnt_area_range(self, bbox, small_kwargs):
        da = generate_burnt_area(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 1.0

    def test_ndvi_with_fire_range(self, bbox, small_kwargs):
        da = generate_ndvi_with_fire(bbox, **small_kwargs)
        assert float(da.min()) >= 0.0
        assert float(da.max()) <= 1.0


# -----------------------------------------------------------------------
# Coordinate tests
# -----------------------------------------------------------------------

class TestCoordinates:
    """Latitude/longitude coordinates should match the bbox."""

    def test_lat_range(self, bbox, small_kwargs):
        da = generate_ndvi(bbox, **small_kwargs)
        lats = da.y.values
        assert lats.min() >= bbox.south
        assert lats.max() <= bbox.north

    def test_lon_range(self, bbox, small_kwargs):
        da = generate_ndvi(bbox, **small_kwargs)
        lons = da.x.values
        assert lons.min() >= bbox.west
        assert lons.max() <= bbox.east

    def test_time_range(self, bbox, small_kwargs):
        da = generate_ndvi(bbox, **small_kwargs)
        assert da.time.values[0] >= np.datetime64(small_kwargs["start"])
        assert da.time.values[-1] < np.datetime64(small_kwargs["end"])


# -----------------------------------------------------------------------
# Attributes
# -----------------------------------------------------------------------

class TestAttributes:

    @pytest.mark.parametrize("gen_fn,expected_name", [
        (generate_ndvi, "NDVI"),
        (generate_lai, "LAI"),
        (generate_fapar, "FAPAR"),
        (generate_fcover, "FCOVER"),
        (generate_gpp, "GPP"),
        (generate_npp, "NPP"),
        (generate_eta, "ETA"),
    ])
    def test_long_name(self, bbox, small_kwargs, gen_fn, expected_name):
        da = gen_fn(bbox, **small_kwargs)
        assert da.attrs["long_name"] == expected_name

    def test_swi_units(self, bbox, small_kwargs):
        da = generate_swi(bbox, **small_kwargs, drought_year=None)
        assert da.attrs["units"] == "%"


# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_same_result(self, bbox, small_kwargs):
        a = generate_ndvi(bbox, **small_kwargs, seed=99)
        b = generate_ndvi(bbox, **small_kwargs, seed=99)
        np.testing.assert_array_equal(a.values, b.values)

    def test_different_seed_different_result(self, bbox, small_kwargs):
        a = generate_ndvi(bbox, **small_kwargs, seed=1)
        b = generate_ndvi(bbox, **small_kwargs, seed=2)
        assert not np.array_equal(a.values, b.values)


# -----------------------------------------------------------------------
# SWI drought
# -----------------------------------------------------------------------

class TestSWIDrought:

    def test_drought_lowers_summer_swi(self, bbox, small_kwargs):
        normal = generate_swi(bbox, **small_kwargs, drought_year=None)
        drought = generate_swi(bbox, **small_kwargs, drought_year=2021)
        # Summer mean should be lower with drought
        normal_mean = float(normal.mean())
        drought_mean = float(drought.mean())
        assert drought_mean < normal_mean


# -----------------------------------------------------------------------
# Burnt area / fire
# -----------------------------------------------------------------------

class TestFireScars:

    def test_fire_creates_scars(self, bbox, small_kwargs):
        fire_times = [(5, 10, 15, 5)]
        ba = generate_burnt_area(bbox, **small_kwargs, fire_times=fire_times)
        assert float(ba.isel(time=5).max()) > 0

    def test_no_fire_all_zero(self, bbox, small_kwargs):
        ba = generate_burnt_area(bbox, **small_kwargs, fire_times=[])
        assert float(ba.max()) == 0.0

    def test_fire_suppresses_ndvi(self, bbox, small_kwargs):
        clean = generate_ndvi(bbox, **small_kwargs, seed=51)
        fire_times = [(5, 10, 15, 5)]
        fired = generate_ndvi_with_fire(bbox, **small_kwargs, seed=51, fire_times=fire_times)
        # At the fire time, NDVI should be lower
        assert float(fired.isel(time=5).mean()) < float(clean.isel(time=5).mean())
