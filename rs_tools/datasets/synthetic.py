"""Synthetic data generators for CGLOPS-like biophysical products.

Provides functions to generate realistic placeholder time-series that
mimic the spatiotemporal patterns of CGLOPS products.  Useful for
notebook demonstrations and unit tests before real data pipelines are
configured.

All generators return ``xarray.DataArray`` objects with ``(time, y, x)``
dimensions and coordinate arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from rs_tools.config import BoundingBox


def _time_axis(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    step_days: int = 10,
) -> np.ndarray:
    """Return a datetime64 array at the given cadence."""
    return np.arange(
        np.datetime64(start),
        np.datetime64(end),
        np.timedelta64(step_days, "D"),
    )


def _day_of_year(times: np.ndarray) -> np.ndarray:
    """Compute fractional day-of-year for each timestamp."""
    return np.array(
        [(t - np.datetime64(str(t)[:4])) / np.timedelta64(1, "D") for t in times],
        dtype=np.float32,
    )


def _seasonal_signal(
    times: np.ndarray,
    phase_shift: float = -np.pi / 2,
) -> np.ndarray:
    """Unit-amplitude seasonal sine wave (peak in ~July for N. hemisphere)."""
    doy = _day_of_year(times)
    return np.sin(2 * np.pi * doy / 365 + phase_shift).astype(np.float32)


def _make_data_array(
    data: np.ndarray,
    times: np.ndarray,
    bbox: BoundingBox,
    name: str,
    units: str,
    ny: int,
    nx: int,
) -> xr.DataArray:
    """Wrap a 3-D array into an xarray.DataArray with proper coords."""
    lats = np.linspace(bbox.north, bbox.south, ny)
    lons = np.linspace(bbox.west, bbox.east, nx)
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": lats, "x": lons},
        attrs={"long_name": name, "units": units},
    )


def _add_noise(
    data: np.ndarray,
    scale: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Add Gaussian noise clipped to the data range."""
    rng = np.random.default_rng(seed)
    noisy = data + rng.normal(0, scale, data.shape).astype(np.float32)
    return noisy


# -----------------------------------------------------------------------
# Public generators
# -----------------------------------------------------------------------

def generate_ndvi(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 42,
) -> xr.DataArray:
    """Synthetic NDVI with latitude-dependent seasonality.

    Higher latitudes show stronger seasonal amplitude; southern latitudes
    remain greener year-round.
    """
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(0.15, 0.5, ny)[None, :, None]
    lat_baseline = np.linspace(0.6, 0.35, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.05, 0.95).astype(np.float32), seed=seed)
    data = np.clip(data, 0.0, 1.0)
    return _make_data_array(data, times, bbox, "NDVI", "-", ny, nx)


def generate_lai(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 43,
) -> xr.DataArray:
    """Synthetic LAI (0–7 m²/m²) correlated with NDVI seasonality."""
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(0.5, 2.5, ny)[None, :, None]
    lat_baseline = np.linspace(4.0, 1.5, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.0, 7.0).astype(np.float32), scale=0.1, seed=seed)
    data = np.clip(data, 0.0, 7.0)
    return _make_data_array(data, times, bbox, "LAI", "m²/m²", ny, nx)


def generate_fapar(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 44,
) -> xr.DataArray:
    """Synthetic FAPAR (0–1) with similar seasonality to NDVI."""
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(0.1, 0.35, ny)[None, :, None]
    lat_baseline = np.linspace(0.55, 0.25, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.0, 1.0).astype(np.float32), seed=seed)
    data = np.clip(data, 0.0, 1.0)
    return _make_data_array(data, times, bbox, "FAPAR", "-", ny, nx)


def generate_fcover(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 45,
) -> xr.DataArray:
    """Synthetic FCOVER (0–1) similar to FAPAR but slightly lower."""
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(0.08, 0.30, ny)[None, :, None]
    lat_baseline = np.linspace(0.50, 0.20, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.0, 1.0).astype(np.float32), seed=seed)
    data = np.clip(data, 0.0, 1.0)
    return _make_data_array(data, times, bbox, "FCOVER", "-", ny, nx)


def generate_gpp(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 46,
) -> xr.DataArray:
    """Synthetic GPP (gC/m²/day) with strong summer peak.

    Gross primary production peaks in summer with values
    typically in the 0–20 gC/m²/day range for European vegetation.
    """
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(2.0, 8.0, ny)[None, :, None]
    lat_baseline = np.linspace(8.0, 3.0, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.0, 20.0).astype(np.float32), scale=0.3, seed=seed)
    data = np.clip(data, 0.0, 20.0)
    return _make_data_array(data, times, bbox, "GPP", "gC/m²/day", ny, nx)


def generate_npp(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 47,
) -> xr.DataArray:
    """Synthetic NPP (gC/m²/day) — roughly half of GPP, can go negative in winter."""
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(1.5, 5.0, ny)[None, :, None]
    lat_baseline = np.linspace(3.0, 0.5, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(data.astype(np.float32), scale=0.3, seed=seed)
    data = np.clip(data, -2.0, 12.0)
    return _make_data_array(data, times, bbox, "NPP", "gC/m²/day", ny, nx)


def generate_eta(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 48,
) -> xr.DataArray:
    """Synthetic actual evapotranspiration (mm/day), peaks in summer."""
    times = _time_axis(start, end, step_days=10)
    seasonal = _seasonal_signal(times)[:, None, None]
    lat_amplitude = np.linspace(0.5, 2.0, ny)[None, :, None]
    lat_baseline = np.linspace(3.0, 1.0, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()
    data = _add_noise(np.clip(data, 0.0, 6.0).astype(np.float32), scale=0.15, seed=seed)
    data = np.clip(data, 0.0, 6.0)
    return _make_data_array(data, times, bbox, "ETA", "mm/day", ny, nx)


def generate_swi(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 49,
    drought_year: Optional[int] = 2022,
) -> xr.DataArray:
    """Synthetic Soil Water Index (0–100%) with optional drought dip.

    If *drought_year* is set, SWI drops sharply during June–September
    of that year.
    """
    times = _time_axis(start, end, step_days=10)
    # SWI is anti-correlated with temperature: wetter in winter
    seasonal = _seasonal_signal(times, phase_shift=np.pi / 2)[:, None, None]
    lat_amplitude = np.linspace(5.0, 20.0, ny)[None, :, None]
    lat_baseline = np.linspace(65.0, 40.0, ny)[None, :, None]

    data = lat_baseline + lat_amplitude * seasonal
    data = np.broadcast_to(data, (len(times), ny, nx)).copy()

    # Inject drought
    if drought_year is not None:
        doy = _day_of_year(times)
        year = np.array([int(str(t)[:4]) for t in times])
        drought_mask = (year == drought_year) & (doy >= 150) & (doy <= 270)
        drought_factor = np.where(drought_mask, 0.55, 1.0)[:, None, None]
        data = data * drought_factor

    data = _add_noise(np.clip(data, 0.0, 100.0).astype(np.float32), scale=2.0, seed=seed)
    data = np.clip(data, 0.0, 100.0)
    return _make_data_array(data, times, bbox, "SWI", "%", ny, nx)


def generate_burnt_area(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 50,
    fire_times: Optional[list] = None,
) -> xr.DataArray:
    """Synthetic burnt-area mask (0/1) with configurable fire events.

    Parameters
    ----------
    fire_times : list of (time_index, y_center, x_center, radius) tuples
        Each tuple places a circular burn scar at the given time step.
        If *None*, two default fires are placed in the southern part of
        the domain during summer months.
    """
    times = _time_axis(start, end, step_days=10)
    data = np.zeros((len(times), ny, nx), dtype=np.float32)

    if fire_times is None:
        # Default: two fire events in summer — southern part of domain
        fire_times = [
            (54, int(ny * 0.8), int(nx * 0.3), 15),   # ~Aug 2021
            (90, int(ny * 0.75), int(nx * 0.6), 20),   # ~Jul 2022
        ]

    yy, xx = np.mgrid[:ny, :nx]
    for t_idx, yc, xc, radius in fire_times:
        if t_idx < len(times):
            dist = np.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
            scar = (dist < radius).astype(np.float32)
            # Scar persists for ~6 dekads (2 months)
            for dt in range(min(6, len(times) - t_idx)):
                fade = max(0.0, 1.0 - dt * 0.15)
                data[t_idx + dt] = np.maximum(data[t_idx + dt], scar * fade)

    return _make_data_array(data, times, bbox, "Burnt Area", "-", ny, nx)


def generate_ndvi_with_fire(
    bbox: BoundingBox,
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    ny: int = 200,
    nx: int = 280,
    seed: int = 51,
    fire_times: Optional[list] = None,
) -> xr.DataArray:
    """NDVI that shows recovery dips at fire scar locations."""
    ndvi = generate_ndvi(bbox, start, end, ny, nx, seed=seed)
    ba = generate_burnt_area(bbox, start, end, ny, nx, seed=seed, fire_times=fire_times)

    # Suppress NDVI where burnt area is active, then slow recovery
    times = ndvi.time.values
    data = ndvi.values.copy()
    ba_vals = ba.values

    for t in range(len(times)):
        if ba_vals[t].max() > 0:
            # Drop NDVI proportional to burn intensity
            data[t] -= ba_vals[t] * 0.4
            # Propagate suppression forward with slow recovery
            for dt in range(1, min(36, len(times) - t)):
                recovery = min(1.0, dt / 36.0)
                suppression = ba_vals[t] * 0.4 * (1.0 - recovery)
                data[t + dt] = np.minimum(data[t + dt], data[t + dt] - suppression)

    data = np.clip(data, 0.0, 1.0)
    ndvi.values = data
    return ndvi
