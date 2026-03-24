# RS-applications

Remote-sensing application toolkit for discovering, accessing, and visualising
Earth-observation data from multiple archives.

## Features

| Capability | Description |
|---|---|
| **Multi-archive search** | Uniform STAC-based access to CDSE, NASA Earthdata, and Terrascope |
| **Dataset catalog** | Pre-registered CLMS, OPERA, and ARIA products with per-archive collection IDs |
| **Data loader** | `load_dataset()` — single call to search, download COGs, and return geo-located xarray objects |
| **RTC composites** | ASF/HyP3 false-colour composites from VV/VH SAR backscatter |
| **GIF animation** | Export time-series as annotated animated GIFs (sensor, orbit, UTC time) |
| **Scale bar** | Geodetic scale bar overlay on imagery plots |
| **Time-series slider** | 2-D map viewer with a date slider |
| **3-D globe inset** | Orthographic globe showing region of interest |
| **Slider comparison** | Before/after or product comparison with draggable divider |
| **RGB compositing** | Multi-temporal or multi-band false-colour composites |

## Repository structure

```
RS-applications/
├── rs_tools/                  # Core Python package
│   ├── config.py              #   Shared search config (bbox, dates, …)
│   ├── search.py              #   Unified multi-archive search
│   ├── archives/              #   Archive connectors
│   │   ├── base.py            #     Abstract base class
│   │   ├── auth.py            #     Terrascope / archive authentication
│   │   ├── cdse.py            #     Copernicus Data Space Ecosystem
│   │   ├── nasa.py            #     NASA Earthdata (CMR STAC)
│   │   └── terrascope.py      #     Terrascope
│   ├── datasets/              #   Known dataset catalog & loader
│   │   ├── catalog.py         #     Dataset registry (CLMS, OPERA, ARIA)
│   │   └── loader.py          #     COG loading → geo-located xarray
│   └── visualization/         #   Reusable visualisation tools
│       ├── animation.py       #     GIF time-series export
│       ├── globe.py           #     3-D globe inset
│       ├── rgb_composite.py   #     RGB multi-temporal compositing
│       ├── rtc_composite.py   #     SAR RTC false-colour composites
│       ├── scalebar.py        #     Geodetic scale bar
│       ├── slider.py          #     Comparison slider plots
│       └── timeseries.py      #     Time-series + slider
├── notebooks/                 # Application Jupyter notebooks
│   └── opera_rtc_timeseries.ipynb
├── tests/                     # Unit tests (pytest)
├── docs/
│   └── credentials.md         # Credential setup guide
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md                  # ← you are here
```

## Quick start

```bash
# Clone
git clone https://github.com/dbekaert/RS-applications.git
cd RS-applications

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Account setup

Searching STAC catalogs is **public** (no credentials needed).
**Downloading** actual data (COG assets) requires free accounts with
the respective archives.

### Terrascope (OPERA RTC, recommended)

1. Register for a free account at <https://terrascope.be/en/sign-up>.
2. Add credentials to `~/.netrc`:

   ```
   machine services.terrascope.be
       login your_email@example.com
       password your_password
   ```

3. `chmod 600 ~/.netrc`
4. In Python, call `setup_terrascope_auth()` before loading data:

   ```python
   from rs_tools.datasets.loader import setup_terrascope_auth
   setup_terrascope_auth()  # reads from ~/.netrc automatically
   ```

   Alternatively, set environment variables `TERRASCOPE_USERNAME` and
   `TERRASCOPE_PASSWORD`.  Tokens are cached and refreshed as needed.

### NASA Earthdata (OPERA RTC, ARIA GUNW)

1. Create a free account at <https://urs.earthdata.nasa.gov/>.
2. Add to `~/.netrc`:

   ```
   machine urs.earthdata.nasa.gov
       login YOUR_USERNAME
       password YOUR_PASSWORD
   ```

3. `chmod 600 ~/.netrc`

### CDSE — Copernicus Data Space Ecosystem

1. Register at <https://dataspace.copernicus.eu/>.
2. Add to `~/.netrc`:

   ```
   machine identity.dataspace.copernicus.eu
       login YOUR_EMAIL
       password YOUR_PASSWORD
   ```

3. `chmod 600 ~/.netrc`

> **Security:** Never commit credentials to version control. Use
> environment variables or a git-ignored `.env` file. See
> [docs/credentials.md](docs/credentials.md) for full details including
> OAuth2 token-based access.

## Known datasets

| Short name | Product | Archive(s) | Resolution |
|---|---|---|---|
| `OPERA_RTC_S1` | OPERA RTC SAR backscatter | Terrascope, NASA | 30 m |
| `OPERA_RTC_S1_STATIC` | OPERA RTC static layers | Terrascope, NASA | 30 m |
| `ARIA_S1_GUNW` | ARIA unwrapped interferograms | NASA | 90 m |
| `CLMS_NDVI_V3` | NDVI (CGLOPS) | CDSE | 300 m |
| `CLMS_LAI_V2` | Leaf Area Index | CDSE | 300 m |
| `CLMS_FAPAR_V2` | Fraction of Absorbed PAR | CDSE | 300 m |
| `CLMS_FCOVER_V2` | Fraction of Vegetation Cover | CDSE | 300 m |
| `CLMS_GPP_V2` | Gross Primary Production | CDSE | 300 m |
| `CLMS_NPP_V2` | Net Primary Production | CDSE | 300 m |
| `CLMS_BA_V4_DAILY` | Burnt Area (daily) | CDSE | 300 m |
| `CLMS_SWI_V4` | Soil Water Index | CDSE | 12.5 km |

```python
from rs_tools.datasets.catalog import list_datasets
for ds in list_datasets():
    print(ds.short_name, ds.name)
```

## Usage example

```python
from rs_tools.config import BoundingBox
from rs_tools.datasets.loader import load_dataset, setup_terrascope_auth

# Authenticate
setup_terrascope_auth()   # reads TERRASCOPE_USERNAME/PASSWORD env vars

# Search + load OPERA RTC data in one call
items = load_dataset(
    "OPERA_RTC_S1",
    bbox=BoundingBox(west=4.35, south=51.21, east=4.42, north=51.25),
    start_date="2024-01-01",
    end_date="2024-06-30",
    archive="terrascope",
    limit=5,
)

# Each item has geo-located VV/VH DataArrays and metadata
for item in items:
    print(item.label)                    # "Sentinel-1A | ASC | 2024-03-15 06:07 UTC"
    print(item.data["VV"].shape)         # (rows, cols) xarray DataArray
    print(item.data["VV"].rio.crs)       # EPSG:32631
```

## Running tests

```bash
pytest -v
```

Offline tests (no network) are the default.  To run integration tests
against live STAC APIs, use:

```bash
pytest -v -m integration
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
