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
from typing import Any, Iterable, List, Optional, Union
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

    def __init__(
        self,
        filename: PathLike,
        max_async_requests: int = DEFAULT_MAX_ASYNC_REQUESTS,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> None:
        """Initialize a remote slide backend with a reusable async session.

        Parameters
        ----------
        filename : PathLike
            URL string representing the remote slide location.
        max_async_requests : int, optional
            Maximum concurrent async requests. Default: 6
        timeout : float, optional
            Request timeout in seconds. Default: 30.0
        **kwargs : Any
            Additional arguments passed to other parent classes.

        Raises
        ------
        ValueError
            If `filename` is a `Path` object or invalid URL format.
        RuntimeError
            If `aiohttp` is not available.
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("`aiohttp` is not available. Install dlup with `remote_backends` dependencies.")

        if isinstance(filename, Path):
            raise ValueError("Filename should be URL string for remote slides.")

        parsed_url = urlparse(filename)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL format: {filename}. Expected format: 'https://example.com/path'")
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only http/https are supported.")

        self._parsed_url: ParseResult = parsed_url
        self._base_url: str = f"{self._parsed_url.scheme}://{self._parsed_url.netloc}"

        # Initialize instance variables and authentication metadata (will be set by _initialize_authentication)
        self._max_async_request = max_async_requests
        self._timeout = timeout
        self.cookies: Optional[dict[str, str]] = None
        self.headers: Optional[dict[str, str]] = None

        # background loop/session state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[aiohttp.ClientSession] = None  # type: ignore
        self._connector: Optional[aiohttp.TCPConnector] = None  # type: ignore
        self._started = threading.Event()
        self._stop_lock = threading.Lock()

        super().__init__(filename)

        # Initialize authentication AFTER parent init (headers/cookies needed for property access)
        self._initialize_authentication()

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of the slide.

        This property is lazy-loaded on first access. The actual fetching is
        performed by the abstract method `_load_slide_metadata()`.
        """
        if not hasattr(self, "_properties"):
            self._properties = self._load_slide_metadata()
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
    def _load_slide_metadata(self) -> dict[str, Any]:
        """Fetch slide-level properties from the remote server.
        Implementations should perform the necessary network request(s) via
        `self.fetch` and return a dictionary of properties.
        """
        pass

    @abstractmethod
    def _initialize_authentication(self) -> None:
        """Initialize authentication and request metadata (headers/cookies).

        This method is called automatically during __init__() AFTER super().__init__()
        returns. This ensures that headers and cookies are set before the calling class
        (in multiple inheritance scenarios) accesses properties that may require
        network requests.

        Implementations should:
        - Parse URL-specific parameters if needed
        - Obtain authentication material (from constructor args or server)
        - Set self.headers and/or self.cookies

        Note: At this point, self._base_url and self._parsed_url are available,
        and parent classes have been initialized. The async session is not yet
        created (that happens lazily on first fetch).
        """
        pass

    # ---------- async workers (run on background loop) ----------
    async def _fetch_single_async(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        cookies: Optional[dict[str, str]] = None,
        retry: int = 2,
        backoff: float = 0.3,
    ) -> bytes:
        """Fetch a single URL using the shared session with simple retries.

        Retries on 429/5xx responses and timeouts with exponential backoff.
        Returns raw bytes.

        Parameters
        ----------
        url : str
            URL to fetch
        method : str
            HTTP method ('GET', 'POST', etc.). Default: 'GET'
        params : dict, optional
            Query parameters (for GET) or form data (for POST)
        data : dict, optional
            Form data for POST requests. For GET requests, if params is None,
            data will be used as params.
        json_data : dict, optional
            JSON body for POST requests
        headers : dict, optional
            Per-request headers (merged with default headers)
        cookies : dict, optional
            Per-request cookies (merged with default cookies)
        retry : int
            Number of retries on failure
        backoff : float
            Base backoff delay in seconds
        """
        if self._session is None:
            raise RuntimeError("Session not initialized. Call _ensure_started() first.")

        # Merge headers/cookies
        request_headers = {**(self.headers or {}), **(headers or {})}
        request_cookies = {**(self.cookies or {}), **(cookies or {})}

        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt <= retry:
            try:
                # Add initial delay before first retry (not before first attempt)
                if attempt > 0:
                    delay = backoff * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)

                if method.upper() == "GET":
                    async with self._session.get(
                        url,
                        params=params,
                        headers=request_headers,
                        cookies=request_cookies,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.read()
                elif method.upper() == "POST":
                    async with self._session.post(
                        url,
                        params=params,
                        data=data,
                        json=json_data,
                        headers=request_headers,
                        cookies=request_cookies,
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.read()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except asyncio.TimeoutError as e:
                last_exc = e
                if attempt == retry:
                    raise RemoteSlideTimeoutError(f"Request timeout after {self._timeout}s for URL: {url}") from e
            except aiohttp.ClientResponseError as e:  # type: ignore
                if e.status in (429, 500, 502, 503, 504) and attempt < retry:
                    last_exc = e
                else:
                    raise RemoteSlideNetworkError(f"HTTP {e.status} for URL {url}: {e.message}") from e
            except aiohttp.ClientError as e:  # type: ignore
                last_exc = e
                if attempt == retry:
                    raise RemoteSlideNetworkError(f"Network error for URL {url}: {e}") from e
            except Exception as e:
                # Unexpected error
                raise RemoteSlideNetworkError(f"Unexpected error for URL {url}: {e}") from e

            attempt += 1

        assert last_exc is not None
        raise RemoteSlideNetworkError(f"Failed for {url}: {last_exc}") from last_exc

    async def _fetch_many_async(
        self,
        urls: Iterable[str],
        params: Optional[Iterable[Optional[dict[str, Any]]]] = None,
        data: Optional[Iterable[Optional[dict[str, Any]]]] = None,
        *,
        concurrency: Optional[int] = None,
    ) -> List[bytes]:
        """Fetch multiple URLs concurrently, bounded by a semaphore.

        Parameters
        ----------
        urls : Iterable[str]
            URLs to fetch
        params : Iterable[dict], optional
            Per-URL query parameters
        data : Iterable[dict], optional
            Per-URL form data (for POST requests)
        concurrency : int, optional
            Maximum concurrent requests. Default: self._max_async_request
        """
        urls = list(urls)
        if params is None:
            params_list: list[Optional[dict[str, Any]]] = [None] * len(urls)
        else:
            params_list = list(params)

        if data is None:
            data_list: list[Optional[dict[str, Any]]] = [None] * len(urls)
        else:
            data_list = list(data)

        if len(urls) != len(params_list):
            raise ValueError("Number of URLs and params must match.")
        if len(urls) != len(data_list):
            raise ValueError("Number of URLs and data must match.")

        sem = asyncio.Semaphore(concurrency or self._max_async_request)

        async def _one(u: str, p: Optional[dict[str, Any]], d: Optional[dict[str, Any]]) -> bytes:
            async with sem:
                return await self._fetch_single_async(u, params=p, data=d)

        tasks = [asyncio.create_task(_one(u, p, d)) for u, p, d in zip(urls, params_list, data_list)]
        results = await asyncio.gather(*tasks)
        return results

    # ---------- public sync API ----------
    def fetch(
        self,
        urls: Union[str, list[str]],
        method: str = "GET",
        params: Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]] = None,
        data: Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]] = None,
    ) -> Union[BytesIO, list[BytesIO]]:
        """Synchronously fetch one or more URLs via the background loop.

        Parameters
        ----------
        urls : Union[str, list[str]]
            Single URL or list of URLs.
        method : str
            HTTP method ('GET', 'POST', etc.). Default: 'GET'
        params : Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]]
            Query parameters (for GET) or form data (for POST).
            For multiple URLs, provide a list of dicts.
        data : Optional[Union[dict[str, Any], list[Optional[dict[str, Any]]]]]
            Form data for POST requests.

        Returns
        -------
        Union[BytesIO, list[BytesIO]]
            Response(s) wrapped in BytesIO for non-async callers.
        """
        self._ensure_started()
        if self._loop is None or not self._thread or not self._thread.is_alive():
            raise RuntimeError(
                "Background thread is not running. Cannot fetch URLs. This may indicate a thread lifecycle issue."
            )

        if isinstance(urls, str):
            if params is not None and not isinstance(params, dict):
                raise ValueError("For single URL, params must be a dict or None")
            if data is not None and not isinstance(data, dict):
                raise ValueError("For single URL, data must be a dict or None")

            fut = asyncio.run_coroutine_threadsafe(
                self._fetch_single_async(urls, method=method, params=params, data=data), self._loop
            )
            b = fut.result()
            return BytesIO(b)
        elif isinstance(urls, list):
            if params is None:
                params_list: list[Optional[dict[str, Any]]] = [None] * len(urls)
            elif isinstance(params, list):
                params_list = params
            else:
                raise ValueError("For multiple URLs, params must be a list of dicts or None")

            if data is None:
                data_list: list[Optional[dict[str, Any]]] = [None] * len(urls)
            elif isinstance(data, list):
                data_list = data
            else:
                raise ValueError("For multiple URLs, data must be a list of dicts or None")

            if len(urls) != len(params_list):
                raise ValueError("Number of URLs and params must match.")
            if len(urls) != len(data_list):
                raise ValueError("Number of URLs and data must match.")

            fut = asyncio.run_coroutine_threadsafe(
                self._fetch_many_async(urls, params=params_list, data=data_list), self._loop
            )
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
        """Thread target: set up loop + session, then run forever until closed so we can reuse ClientSession."""
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
        self._connector = aiohttp.TCPConnector(  # type: ignore
            limit=self._max_async_request,
            limit_per_host=self._max_async_request,
            ttl_dns_cache=300,
            force_close=False,
        )
        timeout = aiohttp.ClientTimeout(total=self._timeout, connect=10, sock_read=self._timeout)  # type: ignore
        self._session = aiohttp.ClientSession(  # type: ignore
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
                await self._connector.close()

    def close(self) -> None:
        """Close session and stop loop/thread."""
        with self._stop_lock:
            if not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(self._shutdown_quiet(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                raise RemoteSlideError(f"Failed to shutdown session and stop loop/thread: {e}") from e
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
