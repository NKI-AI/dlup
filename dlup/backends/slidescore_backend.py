from __future__ import annotations

import functools
import itertools
import json
import math
import os
import pathlib
import re
from io import BytesIO
from typing import Any, Optional

from PIL import Image

import dlup.utils.imports
from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike

if dlup.utils.imports.AIOHTTP_AVAILABLE:
    import asyncio

    import aiohttp

# TODO: Remove magic numbers
METADATA_CACHE = 128
JPEG_QUALITY = 85
SLIDESCORE_MAX_TILE_SIZE = 2000  # SlideScore GetRawTile max size
DEFAULT_API_METHOD = "deepzoom"


def open_slide(filename: PathLike) -> "SlideScoreBackend":
    """
    Read slide with SlideScore backend.

    Parameters
    ----------
    filename : PathLike
        Path to SlideScore image configuration file.
        Should be json format with:
            url
    """
    return SlideScoreBackend(filename)


class SlideScoreBackend(AbstractSlideBackend):
    _max_tile_size = SLIDESCORE_MAX_TILE_SIZE
    _jpeg_quality = JPEG_QUALITY
    _api_method = DEFAULT_API_METHOD

    def __init__(self, filename: PathLike):
        if isinstance(filename, pathlib.Path):
            raise ValueError("Filename should be SlideScore URL for SlideScoreBackend.")
        super().__init__(filename)

        # Parse URL with regex
        parsed_url = re.search(r"(https?://[^/]+).*imageId=(\d+).*studyId=(\d+)", filename)
        if parsed_url is None:
            raise ValueError("Could not parse URL into valid SlideScore parameters.")
        server_url, image_id, study_id = parsed_url.groups()
        self._server_url = server_url
        self._study_id = int(study_id)
        self._image_id = int(image_id)
        self._set_metadata()

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of slide"""
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_properties(self) -> dict[str, Any]:
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetImageMetadata"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        metadata: dict[str, Any] = json.loads(response)["metadata"]
        return metadata

    @property
    def deep_zoom_properties(self) -> dict[str, Any]:
        """Properties of slide"""
        if not hasattr(self, "_deep_zoom_properties"):
            self._deep_zoom_properties = self._fetch_deep_zoom_properties()
        return self._deep_zoom_properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_deep_zoom_properties(self) -> dict[str, Any]:
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetTileServer"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        # Contains JSON object with cookiePart, urlPart (to be used in the calls to /i/ endpoint) and expiresOn
        tile_server: dict[str, Any] = json.loads(response)
        return tile_server

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
        return "RGB"

    def _set_metadata(self) -> None:
        """
        Set up an instance of SlideScore's APIClient for reading images from SlideScore server.
        API token should be experted as `SLIDESCORE_API_TOKEN` in the os environment.

        Parameters
        ----------
        data : dict
            A json containing file information, including slidescore_server_url.

        Returns
        -------
        APIClient
            An instance of the SlideScore APIClient.
        """
        api_token = os.getenv("SLIDESCORE_API_TOKEN")
        if api_token is None:
            raise RuntimeError("SlideScore API token not found. Please set SLIDESCORE_API_TOKEN in os environment")
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}
        self.cookies = None

        # Sanity check for study_id
        response = self.run_fetch_requests(
            urls=[f"{self._server_url}/Api/GetSlideDetails"], data_dicts=[{"imageid": self._image_id}]
        )[0]
        slide_details = json.loads(response)
        if self._study_id != int(slide_details["studyID"]):
            raise RuntimeError(
                f"Slidescore Study ID does not correspond got {slide_details['studyID']} but expected {self._study_id}"
            )

        # Setting properties check `Can get pixels` rights by calling `GetImageMetaData`
        self._level_count = self.properties["levelCount"]
        self._downsamples = self.properties["downsamples"]
        self._spacings = [(float(self.properties["mppX"]), float(self.properties["mppY"]))]
        self._shapes = [(lW, lH) for lW, lH in zip(self.properties["levelWidths"], self.properties["levelHeights"])]

        if self._api_method == "deepzoom":
            self._dz_level_count = math.ceil(
                math.log2(max(self.properties["level0Width"], self.properties["level0Height"]))
            )
            self._dz_tile_size = (self.properties["osdTileSize"], self.properties["osdTileSize"])
            self._dz_stride = (self.properties["osdTileSize"], self.properties["osdTileSize"])
            self.cookies = {"t": self.deep_zoom_properties["cookiePart"]}
        return None

    def read_region(self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]) -> Image.Image:
        """
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
        if self._api_method == "openslide":
            read_fn = self.read_region_openslide
        elif self._api_method == "deepzoom":
            read_fn = self.read_region_deep_zoom
        else:
            raise NotImplementedError("API method of retrieving tiles not implemented.")
        return read_fn(coordinates=coordinates, level=level, size=size)

    def read_region_openslide(self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]) -> Image.Image:
        """
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

        def _create_scaled_range(_coordinate: int, _scale: float, _size: int) -> list[tuple[int, int]]:
            _max_tile_size = math.ceil(_scale * self._max_tile_size)
            _scaled_size = math.ceil(_scale * _size)
            _num_tiles = _scaled_size // _max_tile_size
            _mod_tile_size = _scaled_size % _max_tile_size
            _range = [(_coordinate + _max_tile_size * idx, self._max_tile_size) for idx in range(_num_tiles)]
            # Append risidual tile if modulo is not zero. Scale width / height to appropriate level
            if _mod_tile_size > 0:
                _range.append((_coordinate + _scaled_size - _mod_tile_size, math.ceil(_mod_tile_size / _scale)))
            return _range

        level_downsample = self._downsamples[level]
        x_range = _create_scaled_range(coordinates[0], level_downsample, size[0])
        y_range = _create_scaled_range(coordinates[1], level_downsample, size[1])

        data_dicts = [
            self._request_openslide_data_dict(coordinates=(x, y), level=level, size=(width, height))
            for (y, height), (x, width) in itertools.product(y_range, x_range)
        ]
        tile_urls = [f"{self._server_url}/Api/GetRawTile"] * len(data_dicts)

        tile_responses = self.run_fetch_requests(tile_urls, data_dicts)
        if len(tile_responses) == 1:
            return Image.open(BytesIO(tile_responses[0]))

        _region = Image.new(self.mode, size, (255,) * len(self.mode))
        for tile_data, data in zip(tile_responses, data_dicts):
            _region_tile = Image.open(BytesIO(tile_data))
            x, y = (data["x"], data["y"])
            start_x = math.floor((x - coordinates[0]) / level_downsample)
            start_y = math.floor((y - coordinates[1]) / level_downsample)
            _region.paste(_region_tile, (start_x, start_y))
        return _region

    def read_region_deep_zoom(self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]) -> Image.Image:
        """
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
        level_downsample = round(self._downsamples[level])
        level_dz = self._dz_level_count - round(math.log2(level_downsample))
        level_image_width = math.ceil(self.properties["level0Width"] * level_downsample)
        level_image_height = math.ceil(self.properties["level0Height"] * level_downsample)

        x, y = (coordinates[0] // level_downsample, coordinates[1] // level_downsample)
        w, h = size

        dz_tile_w, dz_tile_h = self._dz_tile_size
        h_stride, v_stride = self._dz_stride

        # Calculate the range of rows and columns for tiles covering the specified region
        start_row = y // dz_tile_h
        end_row = min(math.ceil((y + h) / dz_tile_h), level_image_height // dz_tile_h + 1)
        start_col = x // dz_tile_w
        end_col = min(math.ceil((x + w) / dz_tile_w), level_image_width // dz_tile_w + 1)

        indices = list(itertools.product(range(start_row, end_row), range(start_col, end_col)))
        tile_urls = [self._request_deepzoom_url(level_dz, col, row) for row, col in indices]
        tile_responses = self.run_fetch_requests(tile_urls)

        _region = Image.new(self.mode, size, (255,) * len(self.mode))
        for (row, col), tile_data in zip(indices, tile_responses):
            _region_tile = Image.open(BytesIO(tile_data))
            # Crop tile
            start_y = row * v_stride - y
            end_y = start_y + dz_tile_h
            start_x = col * h_stride - x
            end_x = start_x + dz_tile_w

            img_start_y = max(0, start_y)
            img_end_y = min(h, end_y)
            img_start_x = max(0, start_x)
            img_end_x = min(w, end_x)

            crop_start_y = img_start_y - start_y
            crop_end_y = img_end_y - start_y
            crop_start_x = img_start_x - start_x
            crop_end_x = img_end_x - start_x

            _cropped_region_tile = _region_tile.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
            _region.paste(_cropped_region_tile, (img_start_x, img_start_y))
        return _region

    async def fetch_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        data: Optional[dict[str, Any]],
    ) -> Any:
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
        if data_dicts is None:
            data_dicts = [None] * len(urls)  # type: ignore

        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers) as session:
            tasks = [self.fetch_request(session=session, url=url, data=data) for url, data in zip(urls, data_dicts)]
            return await asyncio.gather(*tasks)

    def run_fetch_requests(
        self,
        urls: list[str],
        data_dicts: Optional[list[dict[str, Any]]] = None,
    ) -> list[Any]:
        return asyncio.get_event_loop().run_until_complete(self.fetch_requests(urls, data_dicts=data_dicts))

    def _request_openslide_data_dict(
        self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]
    ) -> dict[str, Any]:
        """
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
        dict[str, Any]
            The SlideScore API request parameters for a certain region
        """
        # SlideScore images will crash with negative sizes when using GetRawTile for some reason (which is bad)
        assert all(p >= 0 for p in coordinates) and all(p > 0 for p in size) and level >= 0
        return {
            "studyid": self._study_id,
            "imageid": self._image_id,
            "x": coordinates[0],
            "y": coordinates[1],
            "level": level,
            "width": size[0],
            "height": size[1],
            "jpegQuality": self._jpeg_quality,
        }

    def _request_deepzoom_url(self, level: int, col: int, row: int) -> str:
        return (
            f"{self._server_url}/i/{self._image_id}/{self.deep_zoom_properties['urlPart']}"
            f"/i_files/{level}/{col}_{row}.jpeg"
        )

    def close(self) -> None:
        """Close the underlying slide"""
        del self.headers
        del self.cookies
        if self._api_method == "deepzoom":
            del self._deep_zoom_properties
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


if __name__ == "__main__":
    from tqdm import tqdm

    from dlup.data.dataset import TiledWsiDataset

    api_token_path = "/path/to/slide_score_api.key"
    api_token_path = "/Users/ba.d.rooij/Documents/ChangeGamers/azure_devops/api_keys/TCGA.key"
    export_api_key(api_token_path)
    # TCGA breast wsi
    URL_PATH = "https://slidescore.nki.nl/Image/Details?imageId=85162&studyId=638"

    # deepzoom should outperform openslide, but this is not always the case
    for api_method in ["deepzoom", "openslide"]:
        SlideScoreBackend._api_method = api_method
        # Bigger tile_size causes Server disconnected (probably because too many request are made?)
        for tile_size in [(2048, 2048), (256, 256)]:
            dataset = TiledWsiDataset.from_standard_tiling(URL_PATH, 5, tile_size, (0, 0), backend=SlideScoreBackend)

            for tile_region in tqdm(dataset, desc=f"{api_method=}, {tile_size=}"):
                continue
