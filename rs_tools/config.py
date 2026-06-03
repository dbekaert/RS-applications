"""Shared configuration for archive queries.

Provides a SearchConfig dataclass that normalises the common query
parameters (date range, bounding box, product identifiers) accepted by
every archive connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional, Sequence, Tuple, Union


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in EPSG:4326 (lon/lat).

    Attributes
    ----------
    west, south, east, north : float
        Longitude / latitude bounds.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("Longitude must be in [-180, 180].")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("Latitude must be in [-90, 90].")
        if self.south > self.north:
            raise ValueError("south must be <= north.")

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)

    def as_list(self) -> List[float]:
        """Return [west, south, east, north]."""
        return list(self.as_tuple())


def _to_date(value: Union[str, date, datetime]) -> date:
    """Convert a string or datetime to a date object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass
class SearchConfig:
    """Uniform search parameters shared across all archive connectors.

    Parameters
    ----------
    start_date : str | date | datetime
        Start of the temporal search window (inclusive).
    end_date : str | date | datetime
        End of the temporal search window (inclusive).
    bbox : BoundingBox | tuple
        Spatial bounding box.  A 4-tuple is automatically converted to a
        ``BoundingBox(west, south, east, north)``.
    collections : list[str]
        One or more collection / product identifiers recognised by the
        target archive.
    max_cloud_cover : float | None
        Optional maximum cloud-cover percentage (0–100).
    limit : int
        Maximum number of items to return per request.
    """

    start_date: Union[str, date, datetime]
    end_date: Union[str, date, datetime]
    bbox: Union[BoundingBox, Tuple[float, float, float, float]]
    collections: List[str] = field(default_factory=list)
    max_cloud_cover: Optional[float] = None
    limit: int = 100
    include_ea: bool = True  # include Early Adopter collections (NISAR)

    def __post_init__(self) -> None:
        self.start_date = _to_date(self.start_date)
        self.end_date = _to_date(self.end_date)
        if isinstance(self.bbox, (tuple, list)):
            self.bbox = BoundingBox(*self.bbox)
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date.")

    @property
    def date_range_str(self) -> str:
        """ISO date range string ``start/end`` used by STAC APIs."""
        return f"{self.start_date.isoformat()}/{self.end_date.isoformat()}"
