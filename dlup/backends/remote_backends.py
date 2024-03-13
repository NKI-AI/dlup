from .common import AbstractSlideBackend
from abc import abstractmethod


class RemoteSlideBackend(AbstractSlideBackend):
    pass

    @abstractmethod
    def _fetch_metadata(self):
        """Fetch metadata from the remote backend"""
