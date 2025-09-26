from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any

import pyvips

from dlup._types import PathLike
from dlup.backends.deepzoom_backend import AbstractDeepZoomSlide
from dlup.backends.remote_backends import RemoteSlideBackend
from dlup.utils.backends import dict_to_snake_case, parse_xml_to_dict

API_TOKEN_OS_VARIABLE_NAME = "SLIDESCORE_API_TOKEN"


class SlideScoreError(Exception):
    """Base exception for SlideScore backend errors."""
    pass


class SlideScoreAuthenticationError(SlideScoreError):
    """Raised when authentication fails."""
    pass


class SlideScoreNetworkError(SlideScoreError):
    """Raised when network requests fail."""
    pass


def open_slide(filename: PathLike) -> "SlideScoreSlide":
    """
    Read slide with SlideScore backend.

    Parameters
    ----------
    filename : PathLike
        SlideScore URL of format https://<slidescore_server>/Image/Details?imageId=<image_id>&studyId=<study_id>
    max_requests : int, optional
        Maximum number of concurrent requests, by default 6
    """
    return SlideScoreSlide(filename)


class SlideScoreSlide(RemoteSlideBackend, AbstractDeepZoomSlide):
    def __init__(self, filename: PathLike):
        # MRO will take care of setting up metadata. The diamond inheritance is a bit wonky here, but it works.
        # Validation of slidedetails and tileserver properties is done in the set_metadata method
        super().__init__(filename)

    def _fetch_properties(self) -> dict[str, Any]:
        """Fetch properties from GetImageMetadata SlideScore API endpoint"""
        try:
            response = self.fetch(urls=f"{self.base_url}/Api/GetImageMetadata", data={"imageid": self._image_id})
            if not isinstance(response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetImageMetadata endpoint")
            metadata: dict[str, Any] = json.load(response)["metadata"]
            return dict_to_snake_case(metadata)
        except (json.JSONDecodeError, KeyError) as e:
            raise SlideScoreNetworkError(f"Failed to parse metadata response: {e}") from e
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to fetch metadata: {e}") from e

    def _fetch_dz_properties(self) -> dict[str, Any]:
        """Fetch deepzoom properties from GetTileServer SlideScore API endpoint and parse XML response"""
        try:
            dzi_url = f"{self.base_url}/i/{self._image_id}/{self.tile_server_properties['url_part']}/i.dzi"
            dzi_response = self.fetch(urls=dzi_url)
            if not isinstance(dzi_response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from DZI endpoint")
            dz_properties = parse_xml_to_dict(dzi_response)
            return dz_properties
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to fetch DZI properties: {e}") from e

    @property
    def vendor(self) -> str:
        """Returns the scanner vendor."""
        return "SlideScore"

    @property
    def tile_files_root(self) -> str:
        """Returns the SlideScore API endpoint where deep zoom tiles are stored."""
        return f"{self.base_url}/i/{self._image_id}/{self.tile_server_properties['url_part']}/i_files"

    def _set_metadata(self) -> None:
        """Set up metadata for SlideScore server requests and image metadata.
        API token should be exported as `SLIDESCORE_API_TOKEN` in the os environment.

        Raises
        ------
        SlideScoreAuthenticationError
            If SLIDESCORE_API_TOKEN is not an environment variable
        SlideScoreNetworkError
            If serverside studyID is not the same as slide study_id
        """
        api_token = os.getenv(API_TOKEN_OS_VARIABLE_NAME)
        if api_token is None:
            raise SlideScoreAuthenticationError(
                "SlideScore API token not found. Please set SLIDESCORE_API_TOKEN in os environment"
            )
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {api_token}"}

        # Look for studyId and imageId in the URL query (not case sensitive)
        query_match = re.search(r"(?=.*\bstudyId=(\d+))(?=.*\bimageId=(\d+)).*$", self.query, re.IGNORECASE)
        if query_match is None:
            raise ValueError("Could not parse URL into valid SlideScore studyId and imageId.")
        study_id, image_id = query_match.groups()
        self._study_id = int(study_id)
        self._image_id = int(image_id)

        try:
            tile_server_response = self.fetch(urls=f"{self.base_url}/Api/GetTileServer", data={"imageid": self._image_id})
            if not isinstance(tile_server_response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetTileServer endpoint")

            # Contains JSON object with cookiePart, urlPart (to be used in the calls to /i/ endpoint) and expiresOn
            self.tile_server_properties: dict[str, Any] = dict_to_snake_case(json.load(tile_server_response))
            self.cookies = {"t": self.tile_server_properties["cookie_part"]}

            # Validate that we are looking at the correct slide by checking the studyID
            response = self.fetch(urls=f"{self.base_url}/Api/GetSlideDetails", data={"imageid": self._image_id})
            if not isinstance(response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetSlideDetails endpoint")
            
            slide_details = json.load(response)
            if slide_details["studyID"] != self._study_id:
                raise SlideScoreNetworkError("SlideScore Study ID does not correspond to the slide.")
                
        except json.JSONDecodeError as e:
            raise SlideScoreNetworkError(f"Failed to parse tile server response: {e}") from e
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to set up metadata: {e}") from e

    def _fetch_deepzoom_tile_files(self, level: int, indices: list[tuple[int, int]]) -> list[BytesIO]:
        """Retrieve ByteIO objects for tile indices of deepzoom level. The image data will be fetched from the
        SlideScore server in a buffer. The tiles will be fetched asynchronously and stitched together in `read_region`.

        Parameters
        ----------
        level : int
            Deep zoom level for tiles
        indices : list[tuple[int, int]]
            List of (row, col) tuples for row and column at specified deepzoom level

        Returns
        -------
        list[BytesIO]
            List of ByteIO objects for unprocessed DeepZoom tiles.
        """
        try:
            # Super will return list of urls as string
            tile_files_root, file_format = self.tile_files_root, self.dz_properties["image"]["format"]
            tile_urls: list[str] = [f"{tile_files_root}/{level}/{col}_{row}.{file_format}" for row, col in indices]
            tile_responses = self.fetch(tile_urls)
            if not isinstance(tile_responses, list):
                raise SlideScoreNetworkError("Invalid response type from tile fetch")
            return tile_responses
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to fetch deepzoom tiles: {e}") from e

    def _open_deepzoom_tile(self, filename: BytesIO) -> pyvips.Image:
        try:
            return pyvips.Image.new_from_buffer(filename.getvalue(), "")
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to open deepzoom tile: {e}") from e

    def close(self) -> None:
        """Close the underlying slide and clean up resources"""
        super().close()


def export_api_key(file_path: PathLike, os_variable_name: str = API_TOKEN_OS_VARIABLE_NAME) -> None:
    """Reads SlideScore API key from path and exports it into operating system environment.

    Parameters
    ----------
    file_path : PathLike
        Path to the file with SlideScore API key.
    os_variable_name : str, optional
        OS variable to store API key under, by default "SLIDESCORE_API_TOKEN"
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_token = file.read().strip()
        if not api_token:
            raise ValueError("API token file is empty")
        os.environ[os_variable_name] = api_token
    except FileNotFoundError:
        raise SlideScoreAuthenticationError(f"API token file not found: {file_path}")
    except Exception as e:
        raise SlideScoreAuthenticationError(f"Failed to read API token: {e}") from e
