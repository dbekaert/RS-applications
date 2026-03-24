"""Abstract base class for all archive connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from rs_tools.config import SearchConfig


class BaseArchive(ABC):
    """Common interface that every archive connector must implement."""

    name: str = "base"

    @abstractmethod
    def search(self, config: SearchConfig) -> List[Dict[str, Any]]:
        """Search the archive and return a list of item metadata dicts.

        Parameters
        ----------
        config : SearchConfig
            Uniform query parameters.

        Returns
        -------
        list[dict]
            Each dict contains at least ``id``, ``datetime``, ``geometry``,
            and ``assets`` keys.
        """

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return the list of collection identifiers available."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} archive={self.name!r}>"
