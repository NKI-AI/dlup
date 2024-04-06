from __future__ import annotations

import functools
import itertools
import json
import math
import os
import pathlib
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Optional

from PIL import Image

import dlup.utils.imports
from dlup.backends.common import AbstractSlideBackend
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


class SlideScoreSlide(AbstractSlideBackend):
    _max_async_request = DEFAULT_ASYNC_REQUESTS

    def __init__(self, filename: PathLike):
        if isinstance(filename, pathlib.Path):
            raise ValueError("Filename should be SlideScore URL for SlideScoreSlide.")
        super().__init__(filename)

        # Parse URL with regex
        parsed_url = re.search(r"(https?://[^/]+).*imageId=(\d+).*studyId=(\d+)", filename)
        if parsed_url is None:
            raise ValueError("Could not parse URL into valid SlideScore parameters.")
        server_url, image_id, study_id = parsed_url.groups()
        self._server_url = server_url
        self._study_id = int(study_id)
        self._image_id = int(image_id)
        self.cookies: Optional[dict[str, str]] = None
        self.headers: Optional[dict[str, str]] = None
        self._set_metadata()

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of slide"""
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_properties(self) -> dict[str, Any]:
        """Fetch properties from GetImageMetadata SlideScore API endpoint"""
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetImageMetadata"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        metadata: dict[str, Any] = json.loads(response)["metadata"]
        return metadata

    @property
    def dz_properties(self) -> dict[str, Any]:
        """DeepZoom properties of slide"""
        if not hasattr(self, "_dz_properties"):
            self._dz_properties = self._fetch_dz_properties()
        return self._dz_properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_dz_properties(self) -> dict[str, Any]:
        """Fetch deepzoom properties from GetTileServer SlideScore API endpoint and parse XML resonse"""
        dz_properties = {}
        tile_server_response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetTileServer"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        # Contains JSON object with cookiePart, urlPart (to be used in the calls to /i/ endpoint) and expiresOn
        tile_server: dict[str, Any] = json.loads(tile_server_response)
        dz_properties.update(tile_server)

        # Set cookies here first to make deepzoom request
        self.cookies = {"t": dz_properties["cookiePart"]}
        dzi_url = f"{self._server_url}/i/{self._image_id}/{dz_properties['urlPart']}/i.dzi"
        dzi_response = self.run_fetch_requests(urls=[dzi_url])[0]
        dzi_et_root = ET.fromstring(dzi_response)
        dzi_dict = {
            dzi_et_root.tag.split("}")[1]: {k: int(v) if k != "Format" else v for k, v in dzi_et_root.attrib.items()}
        }
        for child in dzi_et_root:
            dzi_dict[dzi_et_root.tag.split("}")[1]][child.tag.split("}")[1]] = {
                k: int(v) for k, v in child.attrib.items()
            }
        dz_properties.update(dzi_dict)
        return dz_properties

    @property
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""
        value = self.properties.get("objectivePower", None)
        if value is not None:
            return int(value)
        return value

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        return "SlideScore"

    @property
    def mode(self) -> str:
        """Returns the mode of the image. This is always RGB for SlideScore"""
        return "RGB"

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
        self.cookies = {"t": self.dz_properties["cookiePart"]}

        # Sanity check for study_id
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetSlideDetails"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        slide_details = json.loads(response)
        if self._study_id != int(slide_details["studyID"]):
            raise RuntimeError(
                f"Slidescore Study ID does not correspond got {slide_details['studyID']} but expected {self._study_id}"
            )

        # Setting properties also checks `Can get pixels` rights by calling `GetImageMetaData`
        self._spacings = [(float(self.properties["mppX"]), float(self.properties["mppY"]))]
        self._level_count = 1 + math.ceil(
            math.log2(
                max(
                    self.dz_properties["Image"]["Size"]["Width"],
                    self.dz_properties["Image"]["Size"]["Height"],
                )
            )
        )
        self._tile_size = (self.dz_properties["Image"]["TileSize"],) * 2
        self._overlap = self.dz_properties["Image"]["Overlap"]

        self._downsamples = [2**level for level in range(self._level_count)]
        self._shapes = [
            (
                math.ceil(self.dz_properties["Image"]["Size"]["Width"] / downsample),
                math.ceil(self.dz_properties["Image"]["Size"]["Height"] / downsample),
            )
            for downsample in self._downsamples
        ]
        self._tile_files = f"{self._server_url}/i/{self._image_id}/{self.dz_properties['urlPart']}/i_files"
        return None

    def read_region(self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]) -> Image.Image:
        """Read region by stitching DeepZoom tiles together.

        Parameters
        ----------
        coordinates : tuple
            Coordinates of the region in level 0.
        level : int
            Level of the image pyramid.
        size : tuple
            Size of the region to be extracted.

        Returns
        -------
        PIL.Image
            The requested region.
        """
        level_downsample = self._downsamples[level]
        level_image_width, level_image_height = self._shapes[level]
        tile_w, tile_h = self._tile_size

        x, y = (coordinates[0] // level_downsample, coordinates[1] // level_downsample)
        w, h = size

        # Calculate the range of rows and columns for tiles covering the specified region
        start_row = y // tile_h
        level_end_row = level_image_height // tile_h + 1
        end_row = min(math.ceil((y + h) / tile_h), level_end_row)
        start_col = x // tile_w
        level_end_col = level_image_width // tile_w + 1
        end_col = min(math.ceil((x + w) / tile_w), level_end_col)

        indices = list(itertools.product(range(start_row, end_row), range(start_col, end_col)))
        level_dz = self._level_count - level - 1
        tile_urls = [self._request_deepzoom_url(level_dz, col, row) for row, col in indices]
        tile_responses = self.run_fetch_requests(tile_urls)

        _region = Image.new(self.mode, size, (255,) * len(self.mode))
        for (row, col), tile_data in zip(indices, tile_responses):
            _region_tile = Image.open(BytesIO(tile_data))
            start_x = col * tile_w - x
            start_y = row * tile_h - y

            img_start_x = max(0, start_x)
            img_end_x = min(w, start_x + tile_w)
            img_start_y = max(0, start_y)
            img_end_y = min(h, start_y + tile_h)

            crop_start_x = img_start_x - start_x
            crop_end_x = img_end_x - start_x
            crop_start_y = img_start_y - start_y
            crop_end_y = img_end_y - start_y

            # Correctly offset for overlap. Edge tile do not have outside padding
            if col > 0:
                crop_start_x += self._overlap
                crop_end_x += self._overlap
            if row > 0:
                crop_start_y += self._overlap
                crop_end_y += self._overlap

            if col == level_end_col:
                crop_end_x -= self._overlap
            if row == level_end_row:
                crop_end_y -= self._overlap
            _cropped_region_tile = _region_tile.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
            _region.paste(_cropped_region_tile, (img_start_x, img_start_y))
        return _region

    async def fetch_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        data: Optional[dict[str, Any]],
    ) -> Any:
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
        Any
            Server response if status == 200 (succes), raises response status otherwise.
        """
        async with session.get(url, data=data) as response:
            if response.status == 200:
                return await response.read()
            else:
                response.raise_for_status()

    async def fetch_requests(
        self,
        urls: list[str],
        data_dicts: Optional[list[dict[str, Any]]] = None,
    ) -> list[Any]:
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
            List of server responses.
        """
        if data_dicts is None:
            data_dicts = [None] * len(urls)  # type: ignore

        connector = aiohttp.TCPConnector(limit=self._max_async_request)
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers, connector=connector) as session:
            tasks = [self.fetch_request(session=session, url=url, data=data) for url, data in zip(urls, data_dicts)]
            return await asyncio.gather(*tasks)

    def run_fetch_requests(
        self,
        urls: list[str],
        data_dicts: Optional[list[dict[str, Any]]] = None,
    ) -> list[Any]:
        """Collect and return all asychronously fetched requests.

        Parameters
        ----------
        urls : list[str]
            List of (base) URLs to get response.
        data_dicts : Optional[list[dict[str, Any]]], optional
            Optionally, add dictionary with data per URL, by default None

        Returns
        -------
        list[Any]
            List of server responses.
        """
        return asyncio.run(self.fetch_requests(urls, data_dicts=data_dicts))

    def _request_deepzoom_url(self, level: int, col: int, row: int) -> str:
        """Generates the url where the JPEG encoded tile is stored

        Parameters
        ----------
        level : int
            Deepzoom level of tile.
        col : int
            Column of requested tile.
        row : int
            Row of requested tile.

        Returns
        -------
        str
            SlideScore URL for tile.
        """
        return f"{self._tile_files}/{level}/{col}_{row}.{self.dz_properties['Image']['Format']}"

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
