"""Unit tests for the backscatter conversion module (offline / no network)."""

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from rs_tools.datasets.backscatter import (
    BackscatterType,
    _ANF_ASSET_KEY,
    apply_anf,
    extract_burst_id,
)


# ── extract_burst_id ────────────────────────────────────────────────────

class TestExtractBurstId:
    def test_rtc_id(self):
        item_id = (
            "OPERA_L2_RTC-S1_T059-124883-IW3_"
            "20240630T174151Z_20240630T194030Z_S1A_30_v1.0"
        )
        assert extract_burst_id(item_id) == "T059-124883-IW3"

    def test_static_id(self):
        item_id = "OPERA_L2_RTC-S1-STATIC_T037-078096-IW3_20140403_S1A_30_v1.0"
        assert extract_burst_id(item_id) == "T037-078096-IW3"

    def test_no_burst(self):
        assert extract_burst_id("some_other_data") is None

    def test_empty(self):
        assert extract_burst_id("") is None


# ── BackscatterType ─────────────────────────────────────────────────────

class TestBackscatterType:
    def test_values(self):
        assert BackscatterType.GAMMA0.value == "gamma0"
        assert BackscatterType.BETA0.value == "beta0"
        assert BackscatterType.SIGMA0.value == "sigma0"

    def test_from_string(self):
        assert BackscatterType("sigma0") == BackscatterType.SIGMA0

    def test_anf_asset_keys(self):
        assert _ANF_ASSET_KEY[BackscatterType.BETA0] == "rtc_anf_gamma0_to_beta0"
        assert _ANF_ASSET_KEY[BackscatterType.SIGMA0] == "rtc_anf_gamma0_to_sigma0"

    def test_gamma0_not_in_anf(self):
        assert BackscatterType.GAMMA0 not in _ANF_ASSET_KEY


# ── apply_anf ───────────────────────────────────────────────────────────

def _make_da(values, crs="EPSG:32631"):
    """Create a small geo-located DataArray for testing."""
    h, w = values.shape
    da = xr.DataArray(
        values.astype(np.float32),
        dims=["y", "x"],
        coords={
            "y": np.arange(h, dtype=np.float64) * 30.0,
            "x": np.arange(w, dtype=np.float64) * 30.0,
        },
    )
    da = da.rio.write_crs(crs)
    from rasterio.transform import from_bounds
    transform = from_bounds(0, 0, w * 30.0, h * 30.0, w, h)
    da = da.rio.write_transform(transform)
    return da


class TestApplyAnf:
    def test_multiplication(self):
        """ANF multiplies gamma-0 to produce the target backscatter."""
        gamma0 = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]]))
        anf = _make_da(np.array([[0.5, 0.5], [2.0, 2.0]]))
        result = apply_anf(gamma0, anf)
        expected = np.array([[0.5, 1.0], [6.0, 8.0]])
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_nan_propagation(self):
        """NaN in gamma-0 stays NaN after conversion."""
        gamma0 = _make_da(np.array([[1.0, np.nan], [3.0, 4.0]]))
        anf = _make_da(np.array([[2.0, 2.0], [2.0, 2.0]]))
        result = apply_anf(gamma0, anf)
        assert np.isnan(result.values[0, 1])
        np.testing.assert_allclose(result.values[0, 0], 2.0)

    def test_anf_nodata_masked(self):
        """Where ANF is 0 (nodata), result should be NaN."""
        gamma0 = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]]))
        anf = _make_da(np.array([[2.0, 0.0], [2.0, 2.0]]))
        result = apply_anf(gamma0, anf)
        assert np.isnan(result.values[0, 1])
        np.testing.assert_allclose(result.values[0, 0], 2.0)

    def test_anf_nan_masked(self):
        """Where ANF is NaN, result should be NaN."""
        gamma0 = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]]))
        anf = _make_da(np.array([[2.0, np.nan], [2.0, 2.0]]))
        result = apply_anf(gamma0, anf)
        assert np.isnan(result.values[0, 1])

    def test_preserves_crs(self):
        """Result should have the same CRS as the input."""
        gamma0 = _make_da(np.array([[1.0, 2.0]]))
        anf = _make_da(np.array([[2.0, 2.0]]))
        result = apply_anf(gamma0, anf)
        assert result.rio.crs is not None

    def test_larger_anf_grid(self):
        """ANF on a larger grid gets reprojected to the RTC grid."""
        gamma0 = _make_da(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # ANF covers a 4x4 area (larger)
        anf_vals = np.full((4, 4), 2.0, dtype=np.float32)
        anf = xr.DataArray(
            anf_vals,
            dims=["y", "x"],
            coords={
                "y": np.arange(4, dtype=np.float64) * 30.0,
                "x": np.arange(4, dtype=np.float64) * 30.0,
            },
        )
        anf = anf.rio.write_crs("EPSG:32631")
        from rasterio.transform import from_bounds
        anf = anf.rio.write_transform(from_bounds(0, 0, 120, 120, 4, 4))

        result = apply_anf(gamma0, anf)
        assert result.shape == gamma0.shape
        np.testing.assert_allclose(result.values, gamma0.values * 2.0, rtol=1e-4)
