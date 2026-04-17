import asyncio
import threading
from io import BytesIO

import fim
import numpy as np
import pytest
from aiohttp import web

from dlup import SlideImage
from dlup.backends.remote_backend import (
    RemoteSlideBackend,
    RemoteSlideNetworkError,
    RemoteSlideTimeoutError,
)


class DummyRemoteBackend(RemoteSlideBackend):
    """Minimal concrete implementation for testing RemoteSlideBackend behaviour."""

    def _initialize_authentication(self):
        """Set basic headers/cookies for the test backend."""
        self.headers = {"X-Test-Header": "1"}
        self.cookies = {"session": "dummy"}

    def _load_slide_metadata(self):
        return {}

    def read_region(self, coordinates, level, size):
        w, h = size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return fim.Image.from_numpy(arr)

    @property
    def magnification(self):
        return None

    @property
    def vendor(self):
        return "DummyRemote"


@pytest.fixture(scope="module")
def aiohttp_test_server():
    """Spin up a simple aiohttp server in a background thread and yield its base URL."""

    async def handle_ok(request: web.Request) -> web.Response:
        return web.Response(text="ok", status=200)

    async def handle_json(request: web.Request) -> web.Response:
        return web.json_response({"message": "hello", "path": request.path})

    async def handle_error(request: web.Request) -> web.Response:
        return web.Response(text="error", status=500)

    async def handle_slow(request: web.Request) -> web.Response:
        await asyncio.sleep(2.0)
        return web.Response(text="slow", status=200)

    state = {"flaky_calls": 0}

    async def handle_flaky(request: web.Request) -> web.Response:
        """Return 500 on first call, 200 on second and later calls."""
        state["flaky_calls"] += 1
        if state["flaky_calls"] == 1:
            return web.Response(text="temporary error", status=500)
        return web.Response(text="ok-after-retry", status=200)

    async def handle_always_error(request: web.Request) -> web.Response:
        """Always return 500 to exercise retry exhaustion."""
        return web.Response(text="always error", status=500)

    app = web.Application()
    app.add_routes(
        [
            web.get("/ok", handle_ok),
            web.get("/json", handle_json),
            web.get("/error", handle_error),
            web.get("/slow", handle_slow),
            web.get("/flaky", handle_flaky),
            web.get("/always_error", handle_always_error),
        ]
    )

    loop = asyncio.new_event_loop()
    runner = None
    site = None
    started = threading.Event()

    def _run() -> None:
        nonlocal runner, site
        asyncio.set_event_loop(loop)
        runner_local = web.AppRunner(app)
        loop.run_until_complete(runner_local.setup())
        site_local = web.TCPSite(runner_local, "127.0.0.1", 0)
        loop.run_until_complete(site_local.start())
        nonlocal runner, site
        runner, site = runner_local, site_local
        started.set()
        try:
            loop.run_forever()
        finally:
            if runner is not None:
                loop.run_until_complete(runner.cleanup())
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    started.wait()

    assert site is not None
    sockets = site._server.sockets  # type: ignore[attr-defined]
    port = sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    try:
        yield base_url
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


class TestRemoteSlideBackend:
    def test_url_parsing_properties(self, aiohttp_test_server: str) -> None:
        url = f"{aiohttp_test_server}/ok?foo=bar"
        backend = DummyRemoteBackend(url)

        assert backend.base_url.startswith("http://127.0.0.1")
        assert backend.path == "/ok"
        assert backend.query == "foo=bar"
        assert backend.hostname == "127.0.0.1"
        assert backend.port is not None

        backend.close()

    def test_fetch_single_ok(self, aiohttp_test_server: str) -> None:
        url = f"{aiohttp_test_server}/ok"
        backend = DummyRemoteBackend(url)

        buf = backend.fetch(url)
        assert buf.read() == b"ok"  # type: ignore[operator]

        backend.close()

    def test_fetch_many_ok(self, aiohttp_test_server: str) -> None:
        url1 = f"{aiohttp_test_server}/ok"
        url2 = f"{aiohttp_test_server}/json"
        backend = DummyRemoteBackend(url1)

        responses = backend.fetch([url1, url2])
        assert isinstance(responses, list)
        first, second = responses
        assert isinstance(first, BytesIO)
        assert isinstance(second, BytesIO)
        assert first.read() == b"ok"  # type: ignore[operator]
        assert b'"message": "hello"' in second.read()  # type: ignore[operator]

        backend.close()

    def test_fetch_http_error_raises_network_error(self, aiohttp_test_server: str) -> None:
        url = f"{aiohttp_test_server}/error"
        backend = DummyRemoteBackend(url)

        with pytest.raises(RemoteSlideNetworkError):
            _ = backend.fetch(url)

        backend.close()

    def test_fetch_timeout_raises_timeout_error(self, aiohttp_test_server: str) -> None:
        # Use a much shorter timeout than the /slow endpoint's 2 seconds.
        url = f"{aiohttp_test_server}/slow"
        backend = DummyRemoteBackend(url, timeout=0.1)

        with pytest.raises(RemoteSlideTimeoutError):
            _ = backend.fetch(url)

        backend.close()

    def test_retry_on_transient_error_succeeds(self, aiohttp_test_server: str) -> None:
        """Server returns 500 once, then 200; fetch should succeed thanks to retries."""
        flaky_url = f"{aiohttp_test_server}/flaky"
        backend = DummyRemoteBackend(flaky_url)

        buf = backend.fetch(flaky_url)
        assert buf.read() == b"ok-after-retry"  # type: ignore[operator]

        backend.close()

    def test_retry_exhaustion_raises_network_error(self, aiohttp_test_server: str) -> None:
        """Server always returns 500; after retries, we should see a RemoteSlideNetworkError."""
        always_error_url = f"{aiohttp_test_server}/always_error"
        backend = DummyRemoteBackend(always_error_url)

        with pytest.raises(RemoteSlideNetworkError):
            _ = backend.fetch(always_error_url)

        backend.close()

    def test_client_error_raises_network_error(self, aiohttp_test_server: str) -> None:
        """Connection errors from aiohttp are mapped to RemoteSlideNetworkError."""
        backend = DummyRemoteBackend(f"{aiohttp_test_server}/ok", timeout=0.5)

        # Use a port that is very likely closed to trigger a connection failure.
        bad_url = "http://127.0.0.1:1"
        with pytest.raises(RemoteSlideNetworkError):
            _ = backend.fetch(bad_url)

        backend.close()

    def test_fetch_many_param_length_mismatch_raises_value_error(self, aiohttp_test_server: str) -> None:
        """Mismatched lengths between urls and params should raise ValueError."""
        url1 = f"{aiohttp_test_server}/ok"
        url2 = f"{aiohttp_test_server}/json"
        backend = DummyRemoteBackend(url1)

        with pytest.raises(ValueError, match="Number of URLs and params must match"):
            _ = backend.fetch([url1, url2], params=[{"a": 1}])

        backend.close()

    def test_explicit_close_stops_background_thread(self, aiohttp_test_server: str) -> None:
        url = f"{aiohttp_test_server}/ok"
        backend = DummyRemoteBackend(url)

        # Trigger background loop/session creation.
        _ = backend.fetch(url)
        assert backend._thread is not None
        thread_ident = backend._thread.ident
        assert thread_ident in {t.ident for t in threading.enumerate()}

        backend.close()

        assert backend._thread is None
        assert thread_ident not in {t.ident for t in threading.enumerate()}

    def test_context_manager_closes_background_thread(self, aiohttp_test_server: str) -> None:
        url = f"{aiohttp_test_server}/ok"

        with DummyRemoteBackend(url) as backend:
            _ = backend.fetch(url)
            assert backend._thread is not None
            thread_ident = backend._thread.ident
            assert thread_ident in {t.ident for t in threading.enumerate()}

        # After context exit, __exit__ should have called close().
        assert backend._thread is None
        assert thread_ident not in {t.ident for t in threading.enumerate()}


class DummyUrlRemoteBackend(RemoteSlideBackend):
    """Tiny RemoteSlideBackend used only to exercise SlideImage.from_file_path URL logic."""

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        self._level_count = 1
        self._downsamples = [1.0]
        self._spacings = [(0.5, 0.5)]
        self._shapes = [(10, 10)]

    def _initialize_authentication(self):
        """No-op authentication initialiser for URL backend tests."""
        self.headers = {}
        self.cookies = {}

    def _load_slide_metadata(self):
        return {}

    def read_region(self, coordinates, level, size):
        w, h = size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return fim.Image.from_numpy(arr)

    @property
    def magnification(self):
        return None

    @property
    def vendor(self):
        return "DummyUrl"


class TestSlideImageFromFilePathUrl:
    def test_url_is_not_converted_to_path_for_remote_backend(self) -> None:
        """Ensure SlideImage.from_file_path accepts URL strings for RemoteSlideBackend subclasses."""
        url = "http://example.com/remote/slide"

        slide_image = SlideImage.from_file_path(url, backend=DummyUrlRemoteBackend)

        backend = slide_image._wsi
        assert isinstance(backend, DummyUrlRemoteBackend)

        filename = backend._filename
        assert isinstance(filename, str)
        assert filename == url
