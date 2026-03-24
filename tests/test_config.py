"""Unit tests for rs_tools.config."""

from datetime import date

import pytest

from rs_tools.config import BoundingBox, SearchConfig


class TestBoundingBox:
    def test_valid_bbox(self):
        bb = BoundingBox(-10, -5, 10, 5)
        assert bb.as_tuple() == (-10, -5, 10, 5)
        assert bb.as_list() == [-10, -5, 10, 5]

    def test_invalid_longitude(self):
        with pytest.raises(ValueError, match="Longitude"):
            BoundingBox(-200, 0, 10, 5)

    def test_invalid_latitude(self):
        with pytest.raises(ValueError, match="Latitude"):
            BoundingBox(0, -100, 10, 5)

    def test_south_greater_than_north(self):
        with pytest.raises(ValueError, match="south must be <= north"):
            BoundingBox(0, 10, 5, -10)


class TestSearchConfig:
    def test_basic_creation(self):
        cfg = SearchConfig(
            start_date="2023-01-01",
            end_date="2023-06-30",
            bbox=(-10, -5, 10, 5),
            collections=["S2_L2A"],
        )
        assert cfg.start_date == date(2023, 1, 1)
        assert cfg.end_date == date(2023, 6, 30)
        assert isinstance(cfg.bbox, BoundingBox)
        assert cfg.date_range_str == "2023-01-01/2023-06-30"

    def test_tuple_bbox_converted(self):
        cfg = SearchConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            bbox=(0, 0, 1, 1),
        )
        assert isinstance(cfg.bbox, BoundingBox)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start_date must be <= end_date"):
            SearchConfig(
                start_date="2024-12-31",
                end_date="2024-01-01",
                bbox=(0, 0, 1, 1),
            )

    def test_default_limit(self):
        cfg = SearchConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            bbox=(0, 0, 1, 1),
        )
        assert cfg.limit == 100
