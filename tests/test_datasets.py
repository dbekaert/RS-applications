"""Unit tests for the dataset catalog."""

import pytest

from rs_tools.datasets.catalog import (
    DatasetInfo,
    get,
    list_datasets,
    register,
)


class TestCatalog:
    def test_clms_products_registered(self):
        expected = [
            "CLMS_NDVI_V3",
            "CLMS_LAI_V2",
            "CLMS_FAPAR_V2",
            "CLMS_FCOVER_V2",
            "CLMS_GPP_V2",
            "CLMS_NPP_V2",
            "CLMS_ETA_V1",
            "CLMS_HF_V1",
            "CLMS_BA_V4_DAILY",
            "CLMS_BA_V4_MONTHLY",
            "CLMS_TOC_V2",
            "CLMS_SWI_V4",
        ]
        for name in expected:
            ds = get(name)
            assert ds.short_name == name

    def test_opera_rtc_registered(self):
        ds = get("OPERA_RTC_S1")
        assert ds.short_name == "OPERA_RTC_S1"
        assert "terrascope" in ds.archive_collections
        assert "nasa" in ds.archive_collections

    def test_opera_rtc_static_registered(self):
        ds = get("OPERA_RTC_S1_STATIC")
        assert "terrascope" in ds.archive_collections
        assert "nasa" in ds.archive_collections

    def test_aria_gunw_registered(self):
        ds = get("ARIA_S1_GUNW")
        assert "nasa" in ds.archive_collections
        assert "insar" in ds.tags

    def test_list_by_tag(self):
        veg = list_datasets(tag="vegetation")
        assert len(veg) >= 3
        for ds in veg:
            assert "vegetation" in ds.tags

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            get("DOES_NOT_EXIST")

    def test_register_custom(self):
        ds = DatasetInfo(
            name="Test Product",
            short_name="TEST_PROD",
            description="A test product.",
            tags=["test"],
        )
        register(ds)
        assert get("TEST_PROD") is ds
