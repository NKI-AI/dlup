from __future__ import annotations

import functools
import json
import os
import pathlib
import re
from io import BytesIO
from typing import Any, Optional

import dlup.utils.imports
from dlup.experimental_backends.deepzoom_backend import (
    DeepZoomSlide,
    TileResponseTypes,
    dict_to_snake_case,
    parse_xml_to_dict,
)
from dlup.types import PathLike

if dlup.utils.imports.AIOHTTP_AVAILABLE:
    import asyncio

    import aiohttp


METADATA_CACHE = 128
DEFAULT_ASYNC_REQUESTS = 6  # This is the number of requests to make asynchronously


def open_slide(filename: PathLike) -> "SlideScoreSlide":
    """
    Read slide with SlideScore backend.

    Parameters
    ----------
    filename : PathLike
        SlideScore URL of format https://<slidescore_server>/Image/Details?imageId=<image_id>&studyId=<study_id>
    """
    return SlideScoreSlide(filename)


class SlideScoreSlide(DeepZoomSlide):
    _max_async_request = DEFAULT_ASYNC_REQUESTS

    def __init__(self, filename: PathLike):
        if isinstance(filename, pathlib.Path):
            raise ValueError("Filename should be SlideScore URL for SlideScoreSlide.")

        # Parse URL with regex
        parsed_url = re.search(r"(https?://[^/?]+)(?=.*\bstudyId=(\d+))(?=.*\bimageId=(\d+)).*$", filename)
        if parsed_url is None:
            raise ValueError("Could not parse URL into valid SlideScore parameters.")
        server_url, study_id, image_id = parsed_url.groups()
        self._server_url = server_url
        self._study_id = int(study_id)
        self._image_id = int(image_id)
        self.cookies: Optional[dict[str, str]] = None
        self.headers: Optional[dict[str, str]] = None
        self._set_metadata()
        super().__init__(filename)

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_properties(self) -> dict[str, Any]:
        """Fetch properties from GetImageMetadata SlideScore API endpoint"""
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetImageMetadata"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        metadata: dict[str, Any] = json.load(response)["metadata"]
        return dict_to_snake_case(metadata)

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_dz_properties(self) -> dict[str, Any]:
        """Fetch deepzoom properties from GetTileServer SlideScore API endpoint and parse XML resonse"""
        dz_properties = {}
        tile_server_response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetTileServer"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        # Contains JSON object with cookiePart, urlPart (to be used in the calls to /i/ endpoint) and expiresOn
        tile_server: dict[str, Any] = json.load(tile_server_response)
        dz_properties.update(dict_to_snake_case(tile_server))

        # Set cookies here to make deepzoom request
        self.cookies = {"t": dz_properties["cookie_part"]}
        dzi_url = f"{self._server_url}/i/{self._image_id}/{dz_properties['url_part']}/i.dzi"
        dzi_response = self.run_fetch_requests(urls=[dzi_url])[0]
        dz_properties.update(parse_xml_to_dict(dzi_response))
        return dz_properties

    @property
    def vendor(self) -> str:
        """Returns the scanner vendor."""
        return "SlideScore"

    @property
    def tile_files(self) -> str:
        """Returns the SlideScore API endpoint where deep zoom tiles are stored."""
        return f"{self._server_url}/i/{self._image_id}/{self.dz_properties['url_part']}/i_files"

    def _set_metadata(self) -> None:
        """Set up metadata for SlideScore server requests and image metadata.
        API token should be experted as `SLIDESCORE_API_TOKEN` in the os environment.

        Raises
        ------
        RuntimeError
            If SLIDESCORE_API_TOKEN is not an environment variable
        RuntimeError
            If serverside studyID is not the same as slide study_id
        """
        api_token = os.getenv("SLIDESCORE_API_TOKEN")
        if api_token is None:
            raise RuntimeError("SlideScore API token not found. Please set SLIDESCORE_API_TOKEN in os environment")
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}
        self.cookies = {"t": self.dz_properties["cookie_part"]}

        # Sanity check for study_id
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetSlideDetails"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        slide_details = json.load(response)
        if self._study_id != int(slide_details["studyID"]):
            raise RuntimeError(
                f"Slidescore Study ID does not correspond got {slide_details['studyID']} but expected {self._study_id}"
            )

    async def fetch_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        data: Optional[dict[str, Any]],
    ) -> BytesIO:
        """Perform a HTTP get request to the url (with extra data)

        Parameters
        ----------
        session : aiohttp.ClientSession
            Interface for making the HTTP requests
        url : str
            URL to perform get request
        data : Optional[dict[str, Any]]
            Additional data for the get request.

        Returns
        -------
        BytesIO
            Server response as a BytesIO object if status == 200 (succes), raises response status otherwise.
        """
        async with session.get(url, data=data) as response:
            if response.status == 200:
                byte_data = await response.read()
                return BytesIO(byte_data)
            else:
                response.raise_for_status()
                raise RuntimeError("Failed to get response from server")  # Raise error for mypy

    async def fetch_requests(
        self,
        urls: list[str],
        data_dicts: Optional[list[dict[str, Any] | None]] = None,
    ) -> list[BytesIO]:
        """Asychronously fetch a list of requests with optional extra data.

        Parameters
        ----------
        urls : list[str]
            List of (base) URLs to get response.
        data_dicts : Optional[list[dict[str, Any]]], optional
            Optionally, add dictionary with data per URL, by default None

        Returns
        -------
        list[Any]
            List of server responses as BytesIO objects.
        """
        if data_dicts is None:
            data_dicts = [None] * len(urls)

        connector = aiohttp.TCPConnector(limit=self._max_async_request)
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers, connector=connector) as session:
            tasks = [self.fetch_request(session=session, url=url, data=data) for url, data in zip(urls, data_dicts)]
            return await asyncio.gather(*tasks)

    def run_fetch_requests(
        self,
        urls: list[str],
        data_dicts: Optional[list[dict[str, Any] | None]] = None,
    ) -> list[Any]:
        """Collect and return all asychronously fetched requests.

        Parameters
        ----------
        urls : list[str]
            List of (base) URLs to get response.
        data_dicts : Optional[list[dict[str, Any] | None]], optional
            Optionally, add dictionary with data per URL, by default None

        Returns
        -------
        list[BytesIO]
            List of server responses as BytesIO objects.
        """
        return asyncio.run(self.fetch_requests(urls, data_dicts=data_dicts))

    def retrieve_deepzoom_tiles(self, level: int, indices: list[tuple[int, int]]) -> list[TileResponseTypes]:
        """Retrieve ByteIO objects for tile indices of deepzoom level. These tiles will be opened with Pillow
        and stitched together in `read_region`. The server requests are made asynchronously.

        Parameters
        ----------
        level : int
            Deep zoom level for tiles
        indices : list[tuple[int, int]]
            List of (col, row) tuples for column and row at specified deepzoom level

        Returns
        -------
        list[BytesIO]
            List of ByteIO objects for unprocessed DeepZoom tiles.
        """
        # Super will return list of urls as string
        tile_urls: list[str] = super().retrieve_deepzoom_tiles(level, indices)  # type: ignore
        tile_responses = self.run_fetch_requests(tile_urls)
        return tile_responses

    def close(self) -> None:
        """Close the underlying slide"""
        del self.headers
        del self.cookies
        del self._dz_properties
        return


def export_api_key(file_path: PathLike, os_variable_name: str = "SLIDESCORE_API_TOKEN") -> None:
    """Reads SlideScore API key from path and exports it into operating system environment.

    Parameters
    ----------
    file_path : PathLike
        Path to the file with SlideScore API key.
    os_variable_name : str, optional
        OS variable to store API key under, by default "SLIDESCORE_API_TOKEN"
    """
    with open(file_path, "r", encoding="utf-8") as file:
        api_token = file.read().strip()
    os.environ[os_variable_name] = api_token
    return
