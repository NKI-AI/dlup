from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import ParseResult, urlparse

from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.imports import AIOHTTP_AVAILABLE

if AIOHTTP_AVAILABLE:
    import asyncio

    import aiohttp


DEFAULT_MAX_ASYNC_REQUESTS = 6  # This is the number of requests to make asynchronously


class RemoteSlideBackend(AbstractSlideBackend):
    """
    Abstract base class for remote slide experimental_backends
    """

    _max_async_request = DEFAULT_MAX_ASYNC_REQUESTS

    def __init__(self, filename: PathLike) -> None:
        """
        Parameters
        ----------
        filename : PathLike
            `PathLike` object representing the URL of the remote slide.

        Raises
        ------
        ValueError
            If filename is not a `str`, but a `Path` object.
        RuntimeError
            If `aiohttp` is not available.
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("`aiohttp` is not available. Install dlup with `remote_backends` dependencies.")

        if isinstance(filename, Path):
            raise ValueError("Filename should be URL string for remote slides.")

        # Some parts of the URL are used frequently, and can be retrieved as properties
        self._parsed_url: ParseResult = urlparse(filename)
        self._base_url: str = f"{self.scheme}://{self.netloc}"  # Combine scheme and netloc for efficient use

        self.cookies: Optional[dict[str, str]] = None
        self.headers: Optional[dict[str, str]] = None
        self._set_metadata()
        super().__init__(filename)

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of the slide. This can be expensive to fetch, so it is set with `self._fetch_properties`."""
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @property
    def scheme(self) -> str:
        """
        Retrieve the scheme (e.g., 'http', 'https') of the parsed URL.
        """
        return self._parsed_url.scheme

    @property
    def netloc(self) -> str:
        """
        Retrieve the network location (e.g., 'example.com:80', '') of the parsed URL.
        """
        return self._parsed_url.netloc

    @property
    def base_url(self) -> str:
        """
        Retrieve the base url (e.g., 'https://example.com:80') consisting of the combined scheme and netloc of the
        parsed URL.
        """
        return self._base_url

    @property
    def path(self) -> str:
        """
        Retrieve the path (e.g., '/path/to/resource', '') of the parsed URL.
        """
        return self._parsed_url.path

    @property
    def query(self) -> str:
        """
        Retrieve the query string (e.g., 'key=value', '') of the parsed URL.
        """
        return self._parsed_url.query

    @property
    def hostname(self) -> Optional[str]:
        """
        Retrieve the hostname (e.g., 'example.com') of the parsed URL.
        """
        return self._parsed_url.hostname

    @property
    def port(self) -> Optional[int]:
        """
        Retrieve the port (e.g., 80, 443) of the parsed URL.
        """
        return self._parsed_url.port

    @abstractmethod
    def _fetch_properties(self) -> dict[str, Any]:
        """Fetch slide properties from the remote server"""
        pass

    @abstractmethod
    def _set_metadata(self) -> None:
        """Metadata needed for remote access"""
        pass

    async def _fetch_single(
        self,
        session: "aiohttp.ClientSession",
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
            Additional data for the get request, by default None.

        Returns
        -------
        BytesIO
            Server response as a BytesIO object if status == 200 (succes), raises response status otherwise.
        """
        async with session.get(url, data=data) as response:
            response.raise_for_status()
            byte_data = await response.read()
            return BytesIO(byte_data)

    async def _fetch_multiple(
        self,
        urls: list[str],
        data_list: Optional[list[Optional[dict[str, Any]]]] = None,
    ) -> list[BytesIO]:
        """Asychronously fetch a list of requests with optional extra data.

        Parameters
        ----------
        urls : list[str]
            A list of URLs to fetch.
        data_list : Optional[list[Optional[dict[str, Any]]]], optional
            A list of dictionaries containing data for each URL, or None for no data.

        Returns
        -------
        list[BytesIO]
            List of server responses as BytesIO objects.
        """
        if data_list is None:
            data_list = [None] * len(urls)

        if len(urls) != len(data_list):
            raise ValueError("Number of URLs and data dictionaries should be the same.")

        connector = aiohttp.TCPConnector(limit=self._max_async_request)
        async with aiohttp.ClientSession(cookies=self.cookies, headers=self.headers, connector=connector) as session:
            tasks = [self._fetch_single(session=session, url=url, data=data) for url, data in zip(urls, data_list)]
            return await asyncio.gather(*tasks)  # pylint: disable=possibly-used-before-assignment

    def fetch(
        self,
        urls: Union[str, list[str]],
        data: Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]] = None,
    ) -> Union[BytesIO, list[BytesIO]]:
        """
        Fetch data from one or more URLs synchronously.

        This method wraps the asynchronous fetching logic in a synchronous interface.

        Parameters
        ----------
        urls : Union[str, list[str]]
            A single URL or a list of URLs to fetch.
        data : Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]]
            A dictionary of data for a single request, or a list of dictionaries for multiple requests.

        Returns
        -------
        Union[BytesIO, List[BytesIO]]
            A BytesIO object for a single URL or a list of BytesIO objects for multiple URLs.
        """
        if isinstance(urls, str) and (isinstance(data, dict) or data is None):
            return asyncio.run(self._fetch_multiple([urls], [data]))[0]
        elif isinstance(urls, list) and (isinstance(data, list) or data is None):
            return asyncio.run(self._fetch_multiple(urls, data))
        else:
            raise ValueError(f"URLs and data should be either both strings or both lists, got {urls=} and {data=}.")
