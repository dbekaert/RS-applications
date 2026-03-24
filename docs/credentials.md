# Credential setup

Each archive requires separate authentication.  **Never commit
credentials to version control.**

---

## NASA Earthdata (`.netrc`)

1. Create a free account at <https://urs.earthdata.nasa.gov/>.
2. Create or edit `~/.netrc`:

   ```
   machine urs.earthdata.nasa.gov
       login YOUR_USERNAME
       password YOUR_PASSWORD
   ```

3. Restrict permissions:

   ```bash
   chmod 600 ~/.netrc
   ```

The `pystac-client` and `requests` libraries will pick up credentials
automatically when accessing NASA CMR STAC endpoints.

---

## Copernicus Data Space Ecosystem (CDSE)

### Option A — `.netrc`

1. Register at <https://dataspace.copernicus.eu/>.
2. Add an entry to `~/.netrc`:

   ```
   machine identity.dataspace.copernicus.eu
       login YOUR_EMAIL
       password YOUR_PASSWORD
   ```

3. Restrict permissions:

   ```bash
   chmod 600 ~/.netrc
   ```

### Option B — OAuth2 access token

For programmatic token-based access (e.g. downloading data), obtain a
token using your CDSE credentials:

```python
import requests

data = {
    "client_id": "cdse-public",
    "grant_type": "password",
    "username": "YOUR_EMAIL",
    "password": "YOUR_PASSWORD",
}

response = requests.post(
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token",
    data=data,
)
access_token = response.json()["access_token"]
```

> **Security note:** Store your credentials in environment variables or
> a `.env` file rather than hard-coding them in scripts.

---

## Terrascope

Terrascope uses OpenID Connect via VITO's identity provider.
Authentication is handled automatically by `setup_terrascope_auth()`
which resolves credentials in this order:

1. Explicit arguments
2. Environment variables
3. `~/.netrc`

### Option A — `.netrc` (recommended)

1. Register at <https://terrascope.be/en/sign-up>.
2. Add an entry to `~/.netrc`:

   ```
   machine services.terrascope.be
       login YOUR_EMAIL
       password YOUR_PASSWORD
   ```

3. Restrict permissions:

   ```bash
   chmod 600 ~/.netrc
   ```

4. In Python:

   ```python
   from rs_tools.archives.auth import login
   login()  # reads from ~/.netrc automatically
   ```

### Option B — Environment variables

```bash
export TERRASCOPE_USERNAME="YOUR_EMAIL"
export TERRASCOPE_PASSWORD="YOUR_PASSWORD"
```

```python
from rs_tools.archives.auth import login
login()  # reads env vars automatically
```

> **Note:** Terrascope uses HTTP Basic Auth via GDAL environment
> variables (`GDAL_HTTP_AUTH`, `GDAL_HTTP_USERPWD`).  No token exchange
> is required — credentials do not expire.

> **Tip:** For interactive use, Terrascope's STAC catalogue
> (`https://stac.terrascope.be/`) is publicly accessible for
> *searching*.  Authentication is only needed for *downloading* assets.

---

## Environment variables (recommended)

To avoid storing passwords in plain text, use environment variables:

```bash
export EARTHDATA_USERNAME="your_user"
export EARTHDATA_PASSWORD="your_pass"
export CDSE_USERNAME="your_email"
export CDSE_PASSWORD="your_pass"
export TERRASCOPE_USERNAME="your_email"
export TERRASCOPE_PASSWORD="your_pass"
```

You can place these in a `.env` file (which is git-ignored) and source
it before running scripts.
