# Copyright (c) dlup contributors
"""Test for the remote abstract backend. Works by creating a mock responses from a server.
TODO: These test should be expanded to cover more uses w.r.t. aiosync and image retrieval. As of right now, it
does not test for any of this, because endpoints on servers will be different. Find a way to do full testing.
"""
from typing import Any

import aiohttp
import pytest
import pyvips
from aioresponses import aioresponses

from dlup.backends.remote_backends import RemoteSlideBackend


class TestRemoteSlideBackend:
    class DummyRemoteSlideBackend(RemoteSlideBackend):
        # Minimal implementation to avoid the ABC restriction.
        def __init__(self, filename: str):
            super().__init__(filename)
            # Dummy data for testing
            self._level_count = 3
            self._downsamples = [1.0, 2.0, 4.0]
            self._spacings = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
            self._shapes = ((1000, 1000), (500, 500), (250, 250))

        def _fetch_properties(self) -> dict[str, Any]:
            return {"property1": "value1", "property2": "value2"}

        def _set_metadata(self) -> None:
            self.metadata = {"meta1": "value1"}

        def read_region(self, coordinates, level, size) -> pyvips.Image:
            return pyvips.Image.black(*size, bands=3)

        @property
        def magnification(self):
            return 10.0

        @property
        def vendor(self):
            return "TestVendor"

        def close(self):
            pass

    @pytest.mark.asyncio
    async def test_fetch_single(self):
        url = "http://mockserver/test"
        data = {"key": "value"}
        response_data = b"response data"

        with aioresponses() as m:
            m.get(url, body=response_data, status=200)

            backend = self.DummyRemoteSlideBackend(url)
            async with aiohttp.ClientSession() as session:
                result = await backend._fetch_single(session, url, data)
                assert result.read() == response_data

    @pytest.mark.asyncio
    async def test_fetch_multiple(self):
        urls = ["http://mockserver/test1", "http://mockserver/test2"]
        data_list = [{"key1": "value1"}, {"key2": "value2"}]
        response_data = [b"response1", b"response2"]

        with aioresponses() as m:
            m.get(urls[0], body=response_data[0], status=200)
            m.get(urls[1], body=response_data[1], status=200)

            backend = self.DummyRemoteSlideBackend(urls[0])
            results = await backend._fetch_multiple(urls, data_list)
            assert [result.read() for result in results] == response_data

    def test_fetch(self):
        url = "http://mockserver/test"
        data = {"key": "value"}
        response_data = b"response data"

        with aioresponses() as m:
            m.get(url, body=response_data, status=200)

            backend = self.DummyRemoteSlideBackend(url)
            result = backend.fetch(url, data)
            assert result.read() == response_data

    def test_properties(self):
        url = "http://mockserver/test"
        backend = self.DummyRemoteSlideBackend(url)
        assert backend.properties == {"property1": "value1", "property2": "value2"}

    def test_metadata(self):
        url = "http://mockserver/test"
        backend = self.DummyRemoteSlideBackend(url)
        assert backend.metadata == {"meta1": "value1"}
