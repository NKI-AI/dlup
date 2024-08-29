from abc import abstractmethod
from typing import Any

from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend


class RemoteSlideBackend(AbstractSlideBackend):
    def __init__(self, filename: PathLike) -> None:
        self._set_metadata()
        super().__init__(filename)

    @property
    def properties(self) -> dict[str, Any]:
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @abstractmethod
    def _fetch_properties(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _set_metadata(self) -> None:
        """Metadata needed for remote access"""
        pass
