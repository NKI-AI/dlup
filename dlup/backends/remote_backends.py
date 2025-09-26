"""
Remote Backend System for DLUP

This module provides a framework for implementing remote slide backends that can work
with DLUP's synchronous dataset interface while internally using async operations for
efficient network I/O.
"""

import asyncio
import threading

from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union, Iterable, List
from urllib.parse import ParseResult, urlparse

from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend
from dlup.utils.imports import AIOHTTP_AVAILABLE

if AIOHTTP_AVAILABLE:
    import aiohttp


DEFAULT_MAX_ASYNC_REQUESTS = 6  # This is the number of requests to make asynchronously
DEFAULT_TIMEOUT = 30.0  # Default timeout in seconds


class RemoteSlideError(Exception):
    """Base exception for remote slide backend errors."""
    pass


class RemoteSlideNetworkError(RemoteSlideError):
    """Raised when network requests fail."""
    pass


class RemoteSlideTimeoutError(RemoteSlideError):
    """Raised when requests timeout."""
    pass


class RemoteSlideBackend(AbstractSlideBackend):
    """
    Abstract base class for remote slide backends

    This class provides a synchronous interface for remote slide access while
    internally using async operations for efficient network I/O.
    """

    _max_async_request = DEFAULT_MAX_ASYNC_REQUESTS
    _timeout = DEFAULT_TIMEOUT

    def __init__(self, filename: PathLike) -> None:
        """Initialize a remote slide backend with a reusable async session.

        Parameters
        ----------
        filename : PathLike
            URL string representing the remote slide location.

        Raises
        ------
        ValueError
            If `filename` is a `Path` object (remote backends expect URL strings).
        RuntimeError
            If `aiohttp` is not available.
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("`aiohttp` is not available. Install dlup with `remote_backends` dependencies.")

        if isinstance(filename, Path):
            raise ValueError("Filename should be URL string for remote slides.")

        # Some parts of the URL are used frequently, and can be retrieved as properties
        self._parsed_url: ParseResult = urlparse(filename)
        self._base_url: str = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}"

        self.cookies: Optional[dict[str, str]] = None
        self.headers: Optional[dict[str, str]] = None

        # background loop/session state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._started = threading.Event()
        self._stop_lock = threading.Lock()

        self._set_metadata()
        super().__init__(filename)

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of the slide. This can be expensive to fetch, so it is set with `self._fetch_properties`."""
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @property
    def parsed_url(self) -> ParseResult:
        """
        Retrieve the parsed URL.
        """
        return self._parsed_url

    @property
    def base_url(self) -> str:
        """
        Retrieve the base url (e.g., 'https://example.com') consisting of the combined scheme and netloc of the
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
        """Fetch slide-level properties from the remote server.

        Implementations should perform the necessary network request(s) via
        `self.fetch` and return a dictionary of properties.
        """
        pass

    @abstractmethod
    def _set_metadata(self) -> None:
        """Populate request metadata (headers/cookies) prior to first request.

        Called during initialization before any network request is issued.
        Implementations can validate the URL and obtain authentication material.
        """
        pass

    # ---------- async workers (run on background loop) ----------
    async def _fetch_single_async(
        self,
        url: str,
        *,
        data: Optional[dict[str, Any]] = None,
        retry: int = 2,
        backoff: float = 0.3,
    ) -> bytes:
        """Fetch a single URL using the shared session with simple retries.

        Retries on 429/5xx responses and timeouts with exponential backoff.
        Returns raw bytes.
        """
        assert self._session is not None
        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= retry:
            try:
                async with self._session.get(
                    url,
                    data=data,  # GET with data
                    headers=self.headers,
                    cookies=self.cookies,
                ) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except asyncio.TimeoutError as e:
                last_exc = e
                if attempt == retry:
                    raise RemoteSlideTimeoutError(f"Request timeout after {self._timeout}s for URL: {url}") from e
            except aiohttp.ClientResponseError as e:
                if e.status in (429, 500, 502, 503, 504) and attempt < retry:
                    last_exc = e
                else:
                    raise RemoteSlideNetworkError(f"HTTP {e.status} for URL {url}: {e.message}") from e
            except aiohttp.ClientError as e:
                last_exc = e
                if attempt == retry:
                    raise RemoteSlideNetworkError(f"Network error for URL {url}: {e}") from e
            except Exception as e:
                # Unexpected
                raise RemoteSlideNetworkError(f"Unexpected error for URL {url}: {e}") from e

            # backoff before retry
            attempt += 1
            await asyncio.sleep(backoff * (2 ** (attempt - 1)))

        assert last_exc is not None
        raise RemoteSlideNetworkError(f"Failed for {url}: {last_exc}") from last_exc

    async def _fetch_many_async(
        self,
        urls: Iterable[str],
        data: Optional[Iterable[Optional[dict[str, Any]]]] = None,
        *,
        concurrency: Optional[int] = None,
    ) -> List[bytes]:
        """Fetch multiple URLs concurrently, bounded by a semaphore."""
        if data is None:
            data = [None] * len(urls)

        # Re-materialize in case caller passed generators
        urls = list(urls)
        data = list(data)
        if len(urls) != len(data):
            raise ValueError("Number of URLs and params must match.")

        sem = asyncio.Semaphore(concurrency or self._max_async_request)

        async def _one(u: str, d: Optional[dict[str, Any]]) -> bytes:
            async with sem:
                return await self._fetch_single_async(
                    u,
                    data=d
                )

        tasks = [asyncio.create_task(_one(u, d)) for u, d in zip(urls, data)]
        results = await asyncio.gather(*tasks)
        return results

    # ---------- public sync API ----------
    def fetch(
        self,
        urls: Union[str, list[str]],
        data: Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]] = None,
    ) -> Union[BytesIO, list[BytesIO]]:
        """Synchronously fetch one or more URLs via the background loop.

        Parameters
        ----------
        urls : Union[str, list[str]]
            Single URL or list of URLs.
        data : Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]]
            Optional per-request query data.

        Returns
        -------
        Union[BytesIO, list[BytesIO]]
            Response(s) wrapped in BytesIO for non-async callers.
        """
        self._ensure_started()
        if isinstance(urls, str):
            data = data if isinstance(data, dict) or data is None else None
            fut = asyncio.run_coroutine_threadsafe(self._fetch_single_async(urls, data=data), self._loop)
            b = fut.result()
            return BytesIO(b)
        elif isinstance(urls, list):
            if data is None:
                data_list = [None] * len(urls)
            elif isinstance(data, list):
                data_list = data
            else:
                raise ValueError("For multiple URLs, data must be a list of params dicts or None")

            fut = asyncio.run_coroutine_threadsafe(self._fetch_many_async(urls, data=data_list), self._loop)
            results = fut.result()
            return [BytesIO(b) for b in results]
        else:
            raise ValueError(f"Unsupported type for urls: {type(urls)}")

    # ---------- lifecycle ----------
    def _ensure_started(self) -> None:
        """Ensure the background event loop and session thread are running."""
        if self._thread and self._thread.is_alive():
            return
        self._started.clear()
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()
        self._started.wait()

    def _run_loop_thread(self) -> None:
        """Thread target: set up loop + session, then run forever until closed."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._create_session())
        self._started.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._shutdown_quiet())
            loop.close()

    async def _create_session(self) -> None:
        """Create a reusable `aiohttp.ClientSession` and connector."""
        self._connector = aiohttp.TCPConnector(
            limit=self._max_async_request,
            limit_per_host=self._max_async_request,
            ttl_dns_cache=300,  # cache DNS
            force_close=False,
        )
        timeout = aiohttp.ClientTimeout(total=self._timeout, connect=10, sock_read=self._timeout)
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            trust_env=True,
        )

    async def _shutdown_quiet(self) -> None:
        """Best-effort async shutdown of session and connector."""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
        finally:
            if self._connector and not self._connector.closed:
                self._connector.close()

    def close(self) -> None:
        """Close session and stop loop/thread."""
        with self._stop_lock:
            if not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(self._shutdown_quiet(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                if self._thread:
                    self._thread.join(timeout=5)
                self._loop = None
                self._thread = None
                self._session = None
                self._connector = None

    def __enter__(self):
        """Context manager entry: start background worker if needed."""
        self._ensure_started()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit: close resources."""
        self.close()

    def __del__(self):
        """Best-effort deleter; explicit close() is preferred"""
        try:
            self.close()
        except Exception:
            pass
