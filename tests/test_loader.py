"""Unit tests for the dataset loader (offline / no network)."""

from datetime import datetime, timezone

import pytest

from rs_tools.datasets.loader import (
    LoadedItem,
    deduplicate_items,
    extract_item_metadata,
    parse_opera_rtc_id,
)


class TestParseOperaRtcId:
    def test_full_id(self):
        item_id = (
            "OPERA_L2_RTC-S1_T059-124883-IW3_"
            "20240630T174151Z_20240630T194030Z_S1A_30_v1.0"
        )
        result = parse_opera_rtc_id(item_id)
        assert result["sensor"] == "S1A"
        assert result["platform"] == "Sentinel-1A"
        assert result["track"] == 59
        assert result["burst"] == 124883
        assert result["swath"] == "IW3"
        assert result["acq_time"] == datetime(2024, 6, 30, 17, 41, 51)
        assert result["proc_time"] == datetime(2024, 6, 30, 19, 40, 30)

    def test_s1b_sensor(self):
        item_id = "OPERA_L2_RTC-S1_T008-015803-IW1_20240627T060717Z_20240629T074105Z_S1B_30_v1.0"
        result = parse_opera_rtc_id(item_id)
        assert result["sensor"] == "S1B"
        assert result["platform"] == "Sentinel-1B"

    def test_no_sensor(self):
        result = parse_opera_rtc_id("some_unknown_id")
        assert "sensor" not in result


class TestDeduplicateItems:
    def _make_item(self, burst, acq, proc):
        item_id = (
            f"OPERA_L2_RTC-S1_T059-{burst:06d}-IW1_"
            f"{acq}Z_{proc}Z_S1A_30_v1.0"
        )
        return {"type": "Feature", "id": item_id, "properties": {}, "assets": {}}

    def test_keeps_latest_proc_time(self):
        items = [
            self._make_item(124883, "20240929T173348", "20240929T210129"),
            self._make_item(124883, "20240929T173348", "20240930T183834"),
        ]
        result = deduplicate_items(items)
        assert len(result) == 1
        assert "20240930T183834" in result[0]["id"]

    def test_no_duplicates_unchanged(self):
        items = [
            self._make_item(124883, "20240929T173348", "20240929T210129"),
            self._make_item(124884, "20240929T173350", "20240929T210130"),
        ]
        result = deduplicate_items(items)
        assert len(result) == 2

    def test_non_opera_items_passed_through(self):
        items = [
            {"id": "generic_item_1", "properties": {}, "assets": {}},
            {"id": "generic_item_2", "properties": {}, "assets": {}},
        ]
        result = deduplicate_items(items)
        assert len(result) == 2

    def test_mixed_opera_and_generic(self):
        items = [
            self._make_item(124883, "20240929T173348", "20240929T210129"),
            self._make_item(124883, "20240929T173348", "20240930T183834"),
            {"id": "generic_item", "properties": {}, "assets": {}},
        ]
        result = deduplicate_items(items)
        assert len(result) == 2  # 1 deduped OPERA + 1 generic

    def test_empty_input(self):
        assert deduplicate_items([]) == []


class TestExtractItemMetadata:
    def test_basic_metadata(self):
        item = {
            "id": "OPERA_L2_RTC-S1_T059-124883-IW3_20240630T174151Z_20240630T194030Z_S1A_30_v1.0",
            "properties": {
                "datetime": "2024-06-30T17:41:51.764886Z",
                "sat:orbit_state": "ascending",
            },
        }
        meta = extract_item_metadata(item)
        assert meta["platform"] == "Sentinel-1A"
        assert meta["orbit_direction"] == "ascending"
        assert meta["datetime"].year == 2024

    def test_missing_orbit(self):
        item = {
            "id": "test_item",
            "properties": {
                "datetime": "2024-01-01T00:00:00Z",
            },
        }
        meta = extract_item_metadata(item)
        assert meta["orbit_direction"] is None


class TestLoadedItem:
    def test_label_full(self):
        item = LoadedItem(
            id="test",
            datetime=datetime(2024, 6, 30, 17, 41, 0, tzinfo=timezone.utc),
            platform="Sentinel-1A",
            orbit_direction="ascending",
        )
        assert item.label == "Sentinel-1A | ASC | 2024-06-30 17:41 UTC"

    def test_label_no_orbit(self):
        item = LoadedItem(
            id="test",
            datetime=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            platform="Sentinel-1B",
        )
        assert item.label == "Sentinel-1B | 2024-01-01 00:00 UTC"
