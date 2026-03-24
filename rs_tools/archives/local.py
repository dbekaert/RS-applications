"""VITO local filesystem access for Terrascope data.

When running on a VITO server, Terrascope STAC items include
``alternate.local.href`` pointing to files on the locally mounted
``/data/MTDA`` network drive (autofs).  Reading from this mount is
dramatically faster than streaming via HTTPS.

Detection mirrors the pattern used in :mod:`rs_tools.archives.s3`
for AWS: a one-time environment check whose result is cached.

Priority order (handled in :func:`rs_tools.datasets.loader.load_stac_asset`):

1. **Local filesystem** — VITO server with ``/data/MTDA`` mount
2. **S3 direct access** — AWS environment with ``/vsis3/``
3. **HTTPS streaming** — ``/vsicurl/`` (universal fallback)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

# Root paths that indicate a VITO server with local data mounts
_VITO_MOUNT_ROOTS = ("/data/MTDA",)

# Module-level detection result (None = not yet checked)
_is_on_vito: Optional[bool] = None

# Cache accessibility per mount subtree so that an unavailable autofs
# sub-mount (e.g. /data/MTDA/NASA) only triggers one slow lookup
# rather than one per asset.
_mount_accessible: Dict[str, bool] = {}


def is_on_vito() -> bool:
    """Detect whether the current environment is a VITO server.

    Checks for the ``/data/MTDA`` autofs infrastructure.  Results
    are cached after the first call.

    Returns
    -------
    bool
    """
    global _is_on_vito  # noqa: PLW0603
    if _is_on_vito is not None:
        return _is_on_vito

    for root in _VITO_MOUNT_ROOTS:
        if os.path.isdir(root):
            logger.info("Running on VITO server (%s mount detected)", root)
            _is_on_vito = True
            return True

    logger.debug("Not running on VITO (mount points not found)")
    _is_on_vito = False
    return False


def _mount_subtree(path: str) -> str:
    """Extract the autofs mount subtree from an absolute path.

    For ``/data/MTDA/NASA/ASF/…`` this returns ``/data/MTDA/NASA``
    (three leading components), which corresponds to the autofs
    indirect map entry that is lazily mounted.
    """
    parts = path.split(os.sep)
    # /data/MTDA/<submount>/…  →  keep first 4 parts (empty, data, MTDA, sub)
    depth = min(4, len(parts))
    return os.sep.join(parts[:depth])


def resolve_local_href(file_url: str) -> Optional[str]:
    """Convert a ``file://`` URL to a local filesystem path.

    Returns the path only when the file actually exists on disk.
    Caches per-subtree accessibility so that a broken autofs
    sub-mount is only probed once.

    Parameters
    ----------
    file_url : str
        A ``file:///data/MTDA/…`` URL from the STAC
        ``alternate.local.href`` field.

    Returns
    -------
    str or None
        Local filesystem path if the file is accessible, ``None``
        otherwise (caller should fall back to HTTPS).
    """
    if not file_url or not file_url.startswith("file://"):
        return None

    local_path = unquote(urlparse(file_url).path)
    subtree = _mount_subtree(local_path)

    # Fast-path: subtree already known to be unavailable
    if _mount_accessible.get(subtree) is False:
        return None

    if os.path.exists(local_path):
        if subtree not in _mount_accessible:
            _mount_accessible[subtree] = True
            logger.info(
                "VITO local mount %s is accessible — using local paths",
                subtree,
            )
        return local_path

    # Distinguish "mount is broken" from "file deleted but mount works"
    if not os.path.isdir(subtree):
        _mount_accessible[subtree] = False
        logger.info(
            "VITO local mount %s not accessible — will use HTTPS",
            subtree,
        )
    else:
        logger.debug("Local path not found (mount OK): %s", local_path)

    return None


def extract_local_url(alternate: Any) -> Optional[str]:
    """Extract the ``file://`` URL from a STAC alternate-assets dict.

    Terrascope items use the `alternate-assets STAC extension`_ with
    structure::

        {"local": {"href": "file:///data/MTDA/…"}}

    Parameters
    ----------
    alternate : dict or str or None
        The ``alternate`` value from a STAC asset entry.

    Returns
    -------
    str or None
        The ``file://`` URL, or ``None`` if not present.

    .. _alternate-assets STAC extension:
       https://github.com/stac-extensions/alternate-assets
    """
    if isinstance(alternate, dict):
        local = alternate.get("local")
        if isinstance(local, dict):
            return local.get("href")
    return None
