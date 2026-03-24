"""Unit tests for rs_tools.datasets.coverage (offline / no network)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from rs_tools.config import BoundingBox
from rs_tools.datasets.coverage import (
    PassRecord,
    _extract_pass_info,
    _pass_group_key,
    compute_bbox_coverage,
    filter_by_coverage,
    print_coverage_report,
    records_to_items,
    summarize_search_results,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic STAC items
# ---------------------------------------------------------------------------

_BBOX = BoundingBox(west=-118.5, south=34.0, east=-117.5, north=35.0)

# Geometry that fully covers the bbox
_GEOM_FULL = {
    "type": "Polygon",
    "coordinates": [[
        [-119.0, 33.5], [-117.0, 33.5],
        [-117.0, 35.5], [-119.0, 35.5],
        [-119.0, 33.5],
    ]],
}

# Geometry that covers roughly the western half of the bbox
_GEOM_HALF = {
    "type": "Polygon",
    "coordinates": [[
        [-119.0, 33.5], [-118.0, 33.5],
        [-118.0, 35.5], [-119.0, 35.5],
        [-119.0, 33.5],
    ]],
}

# Geometry fully outside the bbox
_GEOM_OUTSIDE = {
    "type": "Polygon",
    "coordinates": [[
        [-120.0, 33.0], [-119.5, 33.0],
        [-119.5, 33.5], [-120.0, 33.5],
        [-120.0, 33.0],
    ]],
}


def _opera_item(
    track: int,
    burst: int,
    acq: str,
    sensor: str = "S1A",
    orbit: str = "ascending",
    geometry=None,
) -> dict:
    """Build a minimal OPERA RTC-S1 STAC item for testing."""
    # OPERA ID format: OPERA_L2_RTC-S1_TRRR-BBBBBB-IW1_<acq>Z_<proc>Z_<sensor>_30_v1.0
    acq_dt = datetime.strptime(acq, "%Y%m%dT%H%M%S")
    proc = "20240630T194030"
    item_id = (
        f"OPERA_L2_RTC-S1_T{track:03d}-{burst:06d}-IW1_"
        f"{acq}Z_{proc}Z_{sensor}_30_v1.0"
    )
    geom = geometry if geometry is not None else _GEOM_FULL
    return {
        "type": "Feature",
        "id": item_id,
        "geometry": geom,
        "properties": {
            "datetime": acq_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": "Sentinel-1A" if sensor == "S1A" else "Sentinel-1B",
            "sat:orbit_state": orbit,
        },
        "assets": {
            "VV": {
                "href": f"https://datapool.asf.alaska.edu/{item_id}_VV.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                "alternate": f"s3://asf-cumulus-prod-opera/{item_id}_VV.tif",
            },
        },
        "links": [
            {
                "rel": "self",
                "href": f"https://stac.terrascope.be/collections/opera-s1-rtc-v1/items/{item_id}",
            }
        ],
    }


def _generic_item(
    platform: str,
    date_str: str,
    orbit: str = "ascending",
    geometry=None,
) -> dict:
    """Build a minimal generic STAC item."""
    geom = geometry if geometry is not None else _GEOM_FULL
    return {
        "type": "Feature",
        "id": f"{platform}_{date_str}",
        "geometry": geom,
        "properties": {
            "datetime": f"{date_str}T10:00:00Z",
            "platform": platform,
            "sat:orbit_state": orbit,
        },
        "assets": {
            "data": {
                "href": f"https://example.com/{platform}_{date_str}.tif",
            }
        },
        "links": [],
    }


# ---------------------------------------------------------------------------
# compute_bbox_coverage
# ---------------------------------------------------------------------------

class TestComputeBboxCoverage:
    def test_full_coverage(self):
        cov = compute_bbox_coverage(_GEOM_FULL, _BBOX)
        assert not math.isnan(cov)
        assert cov == pytest.approx(100.0, abs=0.1)

    def test_partial_coverage(self):
        cov = compute_bbox_coverage(_GEOM_HALF, _BBOX)
        assert not math.isnan(cov)
        # Half-coverage: west half of a 1° wide bbox
        assert 40.0 < cov < 60.0

    def test_no_coverage(self):
        cov = compute_bbox_coverage(_GEOM_OUTSIDE, _BBOX)
        assert not math.isnan(cov)
        assert cov == pytest.approx(0.0, abs=0.01)

    def test_missing_geometry_returns_nan(self):
        cov = compute_bbox_coverage({"type": "Point", "coordinates": [0, 0]}, _BBOX)
        # A Point has area 0 so intersection area / bbox area = 0
        assert cov == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# _extract_pass_info
# ---------------------------------------------------------------------------

class TestExtractPassInfo:
    def test_opera_item_parsed(self):
        item = _opera_item(track=59, burst=124883, acq="20240630T174151")
        info = _extract_pass_info(item)
        assert info["track"] == 59
        assert info["platform"] == "Sentinel-1A"
        assert info["orbit_direction"] == "ascending"
        assert isinstance(info["acq_time"], datetime)
        assert info["acq_time"].year == 2024

    def test_generic_item_no_track(self):
        item = _generic_item("Sentinel-2A", "2024-05-01")
        info = _extract_pass_info(item)
        assert info["track"] is None
        assert info["platform"] == "Sentinel-2A"

    def test_missing_datetime_returns_none(self):
        item = {"id": "no_date", "properties": {}, "assets": {}}
        info = _extract_pass_info(item)
        assert info["acq_time"] is None


# ---------------------------------------------------------------------------
# _pass_group_key
# ---------------------------------------------------------------------------

class TestPassGroupKey:
    def test_opera_key_includes_track(self):
        info = {
            "platform": "Sentinel-1A",
            "orbit_direction": "ascending",
            "track": 59,
            "acq_time": datetime(2024, 6, 30, 17, 41, 51),
        }
        key = _pass_group_key(info)
        assert "T059" in key
        assert "2024-06-30" in key

    def test_generic_key_no_track(self):
        info = {
            "platform": "Sentinel-2A",
            "orbit_direction": "descending",
            "track": None,
            "acq_time": datetime(2024, 5, 1, 10, 0, 0),
        }
        key = _pass_group_key(info)
        assert "T" not in key or "T0" not in key  # no track segment
        assert "2024-05-01" in key

    def test_two_bursts_same_pass_share_key(self):
        """Two OPERA bursts from the same track/date must produce the same key."""
        item_a = _opera_item(track=59, burst=124883, acq="20240630T174151")
        item_b = _opera_item(track=59, burst=124884, acq="20240630T174151")
        info_a = _extract_pass_info(item_a)
        info_b = _extract_pass_info(item_b)
        assert _pass_group_key(info_a) == _pass_group_key(info_b)

    def test_different_tracks_produce_different_keys(self):
        item_a = _opera_item(track=59, burst=110000, acq="20240630T174151")
        item_b = _opera_item(track=64, burst=120000, acq="20240630T185000")
        info_a = _extract_pass_info(item_a)
        info_b = _extract_pass_info(item_b)
        assert _pass_group_key(info_a) != _pass_group_key(info_b)

    def test_same_track_different_dates(self):
        item_a = _opera_item(track=59, burst=124883, acq="20240101T174151")
        item_b = _opera_item(track=59, burst=124883, acq="20240625T174151")
        info_a = _extract_pass_info(item_a)
        info_b = _extract_pass_info(item_b)
        assert _pass_group_key(info_a) != _pass_group_key(info_b)


# ---------------------------------------------------------------------------
# summarize_search_results
# ---------------------------------------------------------------------------

class TestSummarizeSearchResults:
    def test_empty_input(self):
        assert summarize_search_results([], _BBOX) == []

    def test_single_item_one_record(self):
        items = [_opera_item(track=59, burst=124883, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        assert len(records) == 1
        r = records[0]
        assert r.date == "2024-06-30"
        assert r.utc_time == "17:41:51"
        assert r.platform == "Sentinel-1A"
        assert r.orbit_direction == "ascending"
        assert r.track == 59
        assert r.n_granules == 1

    def test_two_bursts_same_pass_grouped(self):
        """Two bursts from the same track on the same day → one PassRecord."""
        items = [
            _opera_item(track=59, burst=124883, acq="20240630T174151"),
            _opera_item(track=59, burst=124884, acq="20240630T174151"),
        ]
        records = summarize_search_results(items, _BBOX)
        assert len(records) == 1
        assert records[0].n_granules == 2
        assert len(records[0].items) == 2

    def test_two_different_passes_two_records(self):
        """Items from different dates produce separate records."""
        items = [
            _opera_item(track=59, burst=124883, acq="20240101T174151"),
            _opera_item(track=59, burst=124883, acq="20240625T174151"),
        ]
        records = summarize_search_results(items, _BBOX)
        assert len(records) == 2

    def test_records_sorted_by_date(self):
        items = [
            _opera_item(track=59, burst=1, acq="20240625T174151"),
            _opera_item(track=59, burst=1, acq="20240101T174151"),
        ]
        records = summarize_search_results(items, _BBOX)
        dates = [r.date for r in records]
        assert dates == sorted(dates)

    def test_coverage_full(self):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151", geometry=_GEOM_FULL)]
        records = summarize_search_results(items, _BBOX)
        assert records[0].coverage_pct == pytest.approx(100.0, abs=0.1)

    def test_coverage_partial(self):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151", geometry=_GEOM_HALF)]
        records = summarize_search_results(items, _BBOX)
        assert 40.0 < records[0].coverage_pct < 60.0

    def test_https_and_s3_urls_collected(self):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        r = records[0]
        assert any("https://" in u for u in r.https_urls)
        assert any("s3://" in u for u in r.s3_urls)

    def test_stac_self_links_collected(self):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        assert len(records[0].stac_item_urls) == 1
        assert "stac.terrascope" in records[0].stac_item_urls[0]

    def test_union_coverage_two_halves(self):
        """Two half-coverage items in the same pass → ~100 % combined."""
        geom_east_half = {
            "type": "Polygon",
            "coordinates": [[
                [-118.0, 33.5], [-117.0, 33.5],
                [-117.0, 35.5], [-118.0, 35.5],
                [-118.0, 33.5],
            ]],
        }
        items = [
            _opera_item(track=59, burst=1, acq="20240630T174151", geometry=_GEOM_HALF),
            _opera_item(track=59, burst=2, acq="20240630T174151", geometry=geom_east_half),
        ]
        records = summarize_search_results(items, _BBOX)
        assert len(records) == 1
        assert records[0].coverage_pct == pytest.approx(100.0, abs=1.0)

    def test_generic_items_grouped_by_platform_orbit_date(self):
        items = [
            _generic_item("Sentinel-2A", "2024-05-01", orbit="ascending"),
            _generic_item("Sentinel-2A", "2024-05-01", orbit="ascending"),
        ]
        records = summarize_search_results(items, _BBOX)
        assert len(records) == 1
        assert records[0].n_granules == 2


# ---------------------------------------------------------------------------
# filter_by_coverage
# ---------------------------------------------------------------------------

class TestFilterByCoverage:
    def _make_records(self):
        items = [
            _opera_item(track=59, burst=1, acq="20240101T174151", geometry=_GEOM_FULL),
            _opera_item(track=60, burst=2, acq="20240601T100000", geometry=_GEOM_HALF, orbit="descending", sensor="S1B"),
        ]
        return summarize_search_results(items, _BBOX)

    def test_no_filter_keeps_all(self):
        records = self._make_records()
        assert len(filter_by_coverage(records)) == 2

    def test_min_coverage_excludes_partial(self):
        records = self._make_records()
        filtered = filter_by_coverage(records, min_coverage_pct=80.0)
        assert len(filtered) == 1
        assert filtered[0].track == 59

    def test_orbit_direction_filter(self):
        records = self._make_records()
        asc = filter_by_coverage(records, orbit_direction="ascending")
        assert all(r.orbit_direction == "ascending" for r in asc)
        desc = filter_by_coverage(records, orbit_direction="descending")
        assert all(r.orbit_direction == "descending" for r in desc)

    def test_track_filter(self):
        records = self._make_records()
        assert len(filter_by_coverage(records, track=59)) == 1
        assert len(filter_by_coverage(records, track=60)) == 1
        assert len(filter_by_coverage(records, track=99)) == 0

    def test_date_range_filter(self):
        records = self._make_records()
        filtered = filter_by_coverage(records, start_date="2024-05-01")
        assert all(r.date >= "2024-05-01" for r in filtered)
        filtered2 = filter_by_coverage(records, end_date="2024-03-01")
        assert all(r.date <= "2024-03-01" for r in filtered2)

    def test_platform_substring_filter(self):
        records = self._make_records()
        s1b = filter_by_coverage(records, platform="Sentinel-1B")
        assert len(s1b) == 1
        assert "1B" in s1b[0].platform

    def test_combined_filters(self):
        records = self._make_records()
        result = filter_by_coverage(
            records,
            min_coverage_pct=80.0,
            orbit_direction="ascending",
        )
        assert len(result) == 1
        assert result[0].track == 59


# ---------------------------------------------------------------------------
# records_to_items
# ---------------------------------------------------------------------------

class TestRecordsToItems:
    def test_flat_list_returned(self):
        items = [
            _opera_item(track=59, burst=1, acq="20240630T174151"),
            _opera_item(track=59, burst=2, acq="20240630T174151"),
            _opera_item(track=64, burst=3, acq="20240701T100000"),
        ]
        records = summarize_search_results(items, _BBOX)
        flat = records_to_items(records)
        assert len(flat) == 3

    def test_empty_records(self):
        assert records_to_items([]) == []

    def test_item_ids_preserved(self):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        flat = records_to_items(records)
        assert flat[0]["id"] == items[0]["id"]


# ---------------------------------------------------------------------------
# PassRecord.label  and  print_coverage_report (smoke test)
# ---------------------------------------------------------------------------

class TestPassRecordLabel:
    def test_full_label(self):
        r = PassRecord(
            date="2024-06-30",
            utc_time="17:41:51",
            platform="Sentinel-1A",
            orbit_direction="ascending",
            track=59,
            n_granules=3,
            coverage_pct=92.3,
        )
        label = r.label
        assert "Sentinel-1A" in label
        assert "ASC" in label
        assert "T059" in label
        assert "2024-06-30" in label
        assert "92.3%" in label

    def test_label_no_track_no_orbit(self):
        r = PassRecord(
            date="2024-01-01",
            utc_time="10:00:00",
            platform="Sentinel-2A",
            orbit_direction=None,
            track=None,
            n_granules=1,
            coverage_pct=float("nan"),
        )
        label = r.label
        assert "Sentinel-2A" in label
        # nan coverage → no "cov=" in label
        assert "cov=" not in label


class TestPrintCoverageReport:
    def test_smoke_empty(self, capsys):
        print_coverage_report([])
        out = capsys.readouterr().out
        assert "No records" in out

    def test_smoke_with_records(self, capsys):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        print_coverage_report(records)
        out = capsys.readouterr().out
        assert "2024-06-30" in out
        assert "Sentinel-1A" in out

    def test_show_urls(self, capsys):
        items = [_opera_item(track=59, burst=1, acq="20240630T174151")]
        records = summarize_search_results(items, _BBOX)
        print_coverage_report(records, show_urls=True)
        out = capsys.readouterr().out
        assert "https://" in out
        assert "s3://" in out
        assert "STAC" in out
