from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse

from dlup._types import PathLike
from dlup.backends.deepzoom_backend import AbstractDeepZoomSlide, dict_to_snake_case, parse_xml_to_dict
from dlup.backends.remote_backend import RemoteSlideBackend

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


class SlideScoreSlide(AbstractDeepZoomSlide, RemoteSlideBackend):
    def __init__(
        self,
        filename: PathLike,
        api_token: Optional[str] = None,
        max_async_requests: int = 6,
        timeout: float = 30.0,
    ):
        """Initialize SlideScore slide backend.

        Parameters
        ----------
        filename : PathLike
            SlideScore URL: https://<server>/Image/Details?imageId=<id>&studyId=<id>
        api_token : str, optional
            API token. If not provided, reads from SLIDESCORE_API_TOKEN env var.
        max_async_requests : int, optional
            Maximum concurrent tile requests. Default: 6
        timeout : float, optional
            Request timeout in seconds. Default: 30.0
        """
        # Handle API token; set it before calling super().__init__()
        # _initialize_authentication() will be called during RemoteSlideBackend.__init__() and needs the token
        if api_token is None:
            api_token = os.getenv(API_TOKEN_OS_VARIABLE_NAME)
            if api_token is None:
                raise SlideScoreAuthenticationError(
                    "SlideScore API token not found. Provide api_token parameter "
                    f"or set {API_TOKEN_OS_VARIABLE_NAME} environment variable."
                )
        self._api_token = api_token

        # Parse URL to extract SlideScore-specific IDs (study_id, image_id)
        parsed_url = urlparse(str(filename))
        query_match = re.search(r"(?=.*\bstudyId=(\d+))(?=.*\bimageId=(\d+)).*$", parsed_url.query, re.IGNORECASE)
        if query_match is None:
            raise ValueError(f"Could not parse URL into valid SlideScore studyId and imageId, but got {str(filename)}.")
        study_id, image_id = query_match.groups()
        self._study_id = int(study_id)
        self._image_id = int(image_id)

        # RemoteSlideBackend.__init__() will call _initialize_authentication() after
        # super().__init__() completes, ensuring headers/cookies are ready when
        # AbstractDeepZoomSlide accesses properties during its initialization.
        super().__init__(
            filename,
            max_async_requests=max_async_requests,
            timeout=timeout,
        )

    def _load_slide_metadata(self) -> dict[str, Any]:
        """Fetch properties from GetImageMetadata SlideScore API endpoint"""
        try:
            response = self.fetch(urls=f"{self.base_url}/Api/GetImageMetadata", params={"imageid": self._image_id})
            if not isinstance(response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetImageMetadata endpoint")
            metadata: dict[str, Any] = json.load(response)["metadata"]
            return dict_to_snake_case(metadata)
        except (json.JSONDecodeError, KeyError) as e:
            raise SlideScoreNetworkError(f"Failed to parse metadata response: {e}") from e
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to fetch metadata: {e}") from e

    def _load_dzi_config(self) -> dict[str, Any]:
        """Fetch deepzoom properties from SlideScore DZI endpoint and parse XML response."""
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
        """Returns the SlideScore API endpoint where deep zoom tiles are stored as a string."""
        return f"{self.base_url}/i/{self._image_id}/{self.tile_server_properties['url_part']}/i_files"

    def _validate_study_id(self, slide_details: dict[str, Any]) -> None:
        """Validate that the slide's study ID matches the URL parameter.

        Parameters
        ----------
        slide_details : dict[str, Any]
            Slide details response from GetSlideDetails API endpoint.

        Raises
        ------
        SlideScoreNetworkError
            If the study ID from the server doesn't match the URL parameter.
        """
        if slide_details["studyID"] != self._study_id:
            raise SlideScoreNetworkError(
                f"SlideScore Study ID mismatch: expected {self._study_id}, got {slide_details['studyID']}"
            )

    def _initialize_authentication(self) -> None:
        """Initialize authentication for SlideScore server requests.

        Sets headers and fetches cookies/tile server properties from the server.
        """
        # Set authentication headers
        self.headers = {"Accept": "application/json", "Authorization": f"Bearer {self._api_token}"}

        # Fetch authentication tokens and tile server properties from server
        try:
            tile_server_response = self.fetch(
                urls=f"{self.base_url}/Api/GetTileServer", params={"imageid": self._image_id}
            )
            if not isinstance(tile_server_response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetTileServer endpoint")

            # Contains JSON object with cookiePart, urlPart (to be used in the calls to /i/ endpoint) and expiresOn
            self.tile_server_properties: dict[str, Any] = dict_to_snake_case(json.load(tile_server_response))
            self.cookies = {"t": self.tile_server_properties["cookie_part"]}

            # Validate that we are looking at the correct slide by checking the studyID
            response = self.fetch(urls=f"{self.base_url}/Api/GetSlideDetails", params={"imageid": self._image_id})
            if not isinstance(response, BytesIO):
                raise SlideScoreNetworkError("Invalid response type from GetSlideDetails endpoint")

            slide_details = json.load(response)
            self._validate_study_id(slide_details)

        except json.JSONDecodeError as e:
            raise SlideScoreNetworkError(f"Failed to parse tile server response: {e}") from e
        except SlideScoreNetworkError:
            # Re-raise SlideScoreNetworkError as-is
            raise
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to initialize authentication: {e}") from e

    def _resolve_deepzoom_tile_paths(self, level: int, indices: list[tuple[int, int]]) -> list[BytesIO]:
        """Retrieve ByteIO objects for tile indices of a DeepZoom level from the SlideScore server.

        The image data will be fetched from the SlideScore server in a buffer. The tiles will be fetched
        asynchronously via ``RemoteSlideBackend.fetch`` and stitched together in ``read_region``.

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
        tile_urls = super()._resolve_deepzoom_tile_paths(level, indices)
        try:
            tile_responses = self.fetch(tile_urls)  # type: ignore
            if not isinstance(tile_responses, list):
                raise SlideScoreNetworkError("Invalid response type from tile fetch")
            return tile_responses
        except Exception as e:
            raise SlideScoreNetworkError(f"Failed to fetch deepzoom tiles: {e}") from e

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
