"""AWS S3 direct-access helpers for COG streaming.

When running on an AWS instance, GDAL can access NASA Earthdata
products via ``/vsis3/`` instead of ``/vsicurl/``, bypassing the
OAuth redirect chain and using the AWS internal network.

Detection is performed via:

1. **IMDSv2** — EC2 Instance Metadata Service v2 (standard EC2)
2. **boto3 STS** — ``GetCallerIdentity`` (EKS, ECS, SageMaker,
   Lambda, or any environment with IAM credentials)

Inspired by the ARIA-tools S3 helpers:
https://github.com/aria-tools/ARIA-tools/blob/main/tools/ARIAtools/util/s3.py
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ASF DAAC S3 credential endpoints
_S3_CREDS_ENDPOINTS: Dict[str, str] = {
    "default": "https://cumulus.asf.alaska.edu/s3credentials",
}

# Map S3 bucket prefixes to credential endpoint keys
_BUCKET_TO_ENDPOINT: Dict[str, str] = {}

# Cached credentials (module-level)
_s3_creds_cache: Dict[str, dict] = {}
_s3_creds_expiry: Dict[str, float] = {}

# Module-level detection result (None = not yet checked)
_is_on_aws: Optional[bool] = None


def is_on_aws() -> bool:
    """Detect whether the current environment is running on AWS.

    Uses IMDSv2 first, then falls back to boto3 STS. Results are
    cached after the first call.

    Returns
    -------
    bool
    """
    global _is_on_aws  # noqa: PLW0603
    if _is_on_aws is not None:
        return _is_on_aws

    # Attempt 1: IMDSv2 (fast, no extra dependency)
    try:
        token_resp = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=1.5,
        )
        token_resp.raise_for_status()

        meta_resp = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token_resp.text},
            timeout=1.5,
        )
        meta_resp.raise_for_status()
        logger.info("Running on AWS EC2 (instance %s)", meta_resp.text)
        _is_on_aws = True
        return True
    except Exception:
        logger.debug("IMDSv2 unavailable — trying boto3 fallback")

    # Attempt 2: boto3 STS GetCallerIdentity
    try:
        import boto3
        sts = boto3.client("sts", region_name="us-west-2")
        identity = sts.get_caller_identity()
        logger.info(
            "Running on AWS (STS identity: %s)", identity.get("Arn")
        )
        _is_on_aws = True
        return True
    except ImportError:
        logger.debug("boto3 not installed — cannot use STS fallback")
    except Exception:
        logger.debug("boto3 STS call failed — not on AWS")

    logger.debug("Not running on AWS (all detection methods failed)")
    _is_on_aws = False
    return False


def _endpoint_key_for_s3uri(s3_uri: str) -> str:
    """Return the credential endpoint key for an ``s3://`` URI."""
    bucket = s3_uri[len("s3://"):].split("/", 1)[0]
    return _BUCKET_TO_ENDPOINT.get(bucket, "default")


def _fetch_s3_credentials(endpoint_key: str = "default") -> dict:
    """Fetch temporary S3 credentials from a DAAC endpoint.

    Requires Earthdata Login credentials in ``~/.netrc``.
    """
    url = _S3_CREDS_ENDPOINTS[endpoint_key]
    logger.info("Fetching temporary S3 credentials from %s", url)

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    creds = resp.json()

    required = ("accessKeyId", "secretAccessKey", "sessionToken")
    for key in required:
        if key not in creds:
            raise RuntimeError(
                f"S3 credential response missing key {key!r}"
            )
    return creds


def get_s3_credentials(endpoint_key: str = "default") -> dict:
    """Return cached S3 credentials, refreshing if expired.

    Parameters
    ----------
    endpoint_key : str
        Key into ``_S3_CREDS_ENDPOINTS``.

    Returns
    -------
    dict
        Keys: ``accessKeyId``, ``secretAccessKey``, ``sessionToken``.
    """
    expiry = _s3_creds_expiry.get(endpoint_key, 0)

    # Refresh 5 minutes before expiry
    if endpoint_key not in _s3_creds_cache or time.time() > (expiry - 300):
        _s3_creds_cache[endpoint_key] = _fetch_s3_credentials(endpoint_key)
        _s3_creds_expiry[endpoint_key] = time.time() + 3600

    return _s3_creds_cache[endpoint_key]


def s3uri_to_vsis3(s3_uri: str) -> str:
    """Convert an ``s3://bucket/path`` URI to ``/vsis3/bucket/path``."""
    return "/vsis3/" + s3_uri[len("s3://"):]


def configure_gdal_s3(endpoint_key: str = "default") -> None:
    """Set GDAL config options for S3 direct access.

    Fetches/refreshes temporary credentials from the appropriate DAAC
    endpoint and configures both GDAL config options and environment
    variables (so child processes inherit them).

    Parameters
    ----------
    endpoint_key : str
        Key into ``_S3_CREDS_ENDPOINTS``.
    """
    creds = get_s3_credentials(endpoint_key)

    env_vars = {
        "AWS_ACCESS_KEY_ID": creds["accessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["secretAccessKey"],
        "AWS_SESSION_TOKEN": creds["sessionToken"],
        "AWS_DEFAULT_REGION": "us-west-2",
    }

    # Set as environment variables (works with rasterio/rioxarray)
    for key, value in env_vars.items():
        os.environ[key] = value

    # Also set via GDAL API if available
    try:
        import osgeo.gdal
        osgeo.gdal.SetConfigOption("AWS_ACCESS_KEY_ID", creds["accessKeyId"])
        osgeo.gdal.SetConfigOption(
            "AWS_SECRET_ACCESS_KEY", creds["secretAccessKey"]
        )
        osgeo.gdal.SetConfigOption("AWS_SESSION_TOKEN", creds["sessionToken"])
        osgeo.gdal.SetConfigOption("AWS_REGION", "us-west-2")
        osgeo.gdal.SetConfigOption("AWS_NO_SIGN_REQUEST", "NO")
    except ImportError:
        pass

    logger.info(
        "GDAL configured for S3 direct access (endpoint: %s)", endpoint_key
    )


def resolve_href(
    https_url: str,
    s3_url: Optional[str] = None,
) -> str:
    """Choose the best GDAL virtual filesystem path for a COG.

    On AWS with an S3 URL available, returns ``/vsis3/...``.
    Otherwise returns ``/vsicurl/<https_url>`` for streaming.

    Parameters
    ----------
    https_url : str
        HTTPS download URL.
    s3_url : str, optional
        Corresponding ``s3://`` URI if available.

    Returns
    -------
    str
        GDAL-ready path (``/vsis3/...`` or ``/vsicurl/...``).
    """
    if s3_url and is_on_aws():
        endpoint_key = _endpoint_key_for_s3uri(s3_url)
        configure_gdal_s3(endpoint_key)
        return s3uri_to_vsis3(s3_url)

    return f"/vsicurl/{https_url}"
