"""Unit tests for archive connectors (offline / mocked)."""

import pytest

from rs_tools.archives.base import BaseArchive
from rs_tools.config import SearchConfig
from rs_tools.search import get_archive, _ARCHIVE_REGISTRY


class TestArchiveRegistry:
    def test_known_archives(self):
        for name in ("cdse", "nasa", "terrascope"):
            assert name in _ARCHIVE_REGISTRY

    def test_get_archive_valid(self):
        # We can't connect to the real APIs in CI, so just verify the
        # registry look-up returns the right type.
        for name, cls in _ARCHIVE_REGISTRY.items():
            assert issubclass(cls, BaseArchive)

    def test_get_archive_unknown(self):
        with pytest.raises(ValueError, match="Unknown archive"):
            get_archive("nonexistent")


class TestBaseArchive:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseArchive()
