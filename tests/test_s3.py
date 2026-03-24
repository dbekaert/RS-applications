"""Unit tests for the S3 direct-access helpers (offline / mocked)."""

from unittest.mock import patch

import pytest

from rs_tools.archives.s3 import (
    is_on_aws,
    resolve_href,
    s3uri_to_vsis3,
    _endpoint_key_for_s3uri,
)


class TestS3UriConversion:
    def test_basic_uri(self):
        result = s3uri_to_vsis3("s3://my-bucket/path/to/file.tif")
        assert result == "/vsis3/my-bucket/path/to/file.tif"

    def test_deep_path(self):
        result = s3uri_to_vsis3("s3://bucket/a/b/c/d.tif")
        assert result == "/vsis3/bucket/a/b/c/d.tif"


class TestEndpointKeyForS3Uri:
    def test_default_bucket(self):
        key = _endpoint_key_for_s3uri("s3://unknown-bucket/path/file.tif")
        assert key == "default"


class TestIsOnAws:
    @patch("rs_tools.archives.s3.requests.put", side_effect=Exception("no IMDS"))
    def test_not_on_aws_no_boto(self, mock_put):
        """When IMDSv2 fails and boto3 is not available, returns False."""
        import rs_tools.archives.s3 as s3_mod
        s3_mod._is_on_aws = None  # Reset cache
        with patch.dict("sys.modules", {"boto3": None}):
            # Force reimport failure for boto3
            result = is_on_aws()
        # Restore cache
        s3_mod._is_on_aws = None
        assert result is False


class TestResolveHref:
    def test_no_s3_url_returns_vsicurl(self):
        """Without S3 URL, should return a vsicurl path."""
        result = resolve_href("https://example.com/data.tif")
        assert result == "/vsicurl/https://example.com/data.tif"

    def test_s3_url_but_not_on_aws(self):
        """With S3 URL but not on AWS, should fallback to vsicurl."""
        import rs_tools.archives.s3 as s3_mod
        s3_mod._is_on_aws = False  # Force not on AWS
        try:
            result = resolve_href(
                "https://example.com/data.tif",
                s3_url="s3://bucket/data.tif",
            )
            assert result == "/vsicurl/https://example.com/data.tif"
        finally:
            s3_mod._is_on_aws = None  # Reset


class TestResolveNasaHref:
    def test_import(self):
        """resolve_nasa_href should be importable."""
        from rs_tools.archives.nasa import resolve_nasa_href
        assert callable(resolve_nasa_href)
