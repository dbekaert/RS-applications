"""Terrascope authentication via HTTP Basic Auth.

Configures GDAL environment variables so that ``rasterio`` / GDAL can
stream Cloud-Optimized GeoTIFFs from the Terrascope download service.

Uses the same mechanism as the `leafmap.terrascope` module:
``GDAL_HTTP_AUTH=BASIC`` + ``GDAL_HTTP_USERPWD=user:pass``.

Credentials are resolved in order:

1. Explicit ``username`` / ``password`` arguments
2. ``TERRASCOPE_USERNAME`` / ``TERRASCOPE_PASSWORD`` environment variables
3. ``~/.netrc`` entry for ``services.terrascope.be``
"""

from __future__ import annotations

import logging
import netrc
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_NETRC_MACHINE = "services.terrascope.be"
_logged_in = False


def _read_netrc() -> Tuple[Optional[str], Optional[str]]:
    """Try to read Terrascope credentials from ``~/.netrc``."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return None, None
    try:
        info = netrc.netrc(str(netrc_path))
        auth = info.authenticators(_NETRC_MACHINE)
        if auth:
            return auth[0], auth[2]  # (login, password)
    except netrc.NetrcParseError:
        logger.warning("Failed to parse ~/.netrc")
    return None, None


def _resolve_credentials(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve Terrascope credentials from multiple sources."""
    username = username or os.environ.get("TERRASCOPE_USERNAME")
    password = password or os.environ.get("TERRASCOPE_PASSWORD")

    if not username or not password:
        netrc_user, netrc_pass = _read_netrc()
        username = username or netrc_user
        password = password or netrc_pass

    if not username or not password:
        raise RuntimeError(
            "Terrascope credentials not found. Provide them via:\n"
            "  1) username/password arguments, or\n"
            "  2) TERRASCOPE_USERNAME / TERRASCOPE_PASSWORD env vars, or\n"
            "  3) ~/.netrc entry for machine services.terrascope.be"
        )
    return username, password


def login(
    username: Optional[str] = None,
    password: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """Authenticate with Terrascope using HTTP Basic Auth.

    Sets GDAL environment variables so that ``rasterio`` / GDAL can
    access authenticated Terrascope COG downloads.  No token exchange
    is needed — basic auth credentials do not expire.

    Parameters
    ----------
    username, password : str, optional
        Terrascope credentials.  Falls back to env vars and ``~/.netrc``.
    quiet : bool
        Suppress status messages.
    """
    global _logged_in  # noqa: PLW0603

    username, password = _resolve_credentials(username, password)

    os.environ["GDAL_HTTP_AUTH"] = "BASIC"
    os.environ["GDAL_HTTP_USERPWD"] = f"{username}:{password}"
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    _logged_in = True

    if not quiet:
        print(f"Authenticated as: {username}")


def logout() -> None:
    """Clear GDAL authentication configuration."""
    global _logged_in  # noqa: PLW0603

    for var in [
        "GDAL_HTTP_AUTH",
        "GDAL_HTTP_USERPWD",
        "GDAL_DISABLE_READDIR_ON_OPEN",
        "GDAL_HTTP_HEADERS",
    ]:
        os.environ.pop(var, None)

    _logged_in = False
    print("Logged out from Terrascope")


def is_logged_in() -> bool:
    """Return True if Terrascope credentials are configured."""
    return _logged_in
