import asyncio
import json
import threading
from io import BytesIO
from typing import Any, Optional

import fim
import numpy as np
import PIL.Image
import pytest
from aiohttp import web

from dlup import SlideImage
from dlup.backends.slidescore_backend import (
    API_TOKEN_OS_VARIABLE_NAME,
    SlideScoreAuthenticationError,
    SlideScoreNetworkError,
    SlideScoreSlide,
)
from dlup.utils.backends import ImageBackend


def _write_dzi_xml_for_slidescore(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    fmt: str = "jpeg",
) -> bytes:  # noqa: D401
    """Create a minimal DeepZoom XML description matching the OpenSlide DeepZoom format."""
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}"
       xmlns="http://schemas.microsoft.com/deepzoom/2008">
  <Size Width="{width}" Height="{height}"/>
</Image>
"""
    return content.encode("utf-8")


def _make_tile(color: tuple[int, int, int]) -> bytes:
    img = PIL.Image.fromarray(np.full((2, 2, 3), color, dtype=np.uint8), mode="RGB")
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture(scope="module")
def slidescore_server():
    """Start a mocked SlideScore server exposing the minimal API surface for SlideScoreSlide."""
    from dlup.utils.imports import AIOHTTP_AVAILABLE

    if not AIOHTTP_AVAILABLE:
        pytest.skip("aiohttp not available, skipping SlideScore backend tests.")

    width, height = 4, 4
    tile_size, overlap = 2, 0
    dzi_bytes = _write_dzi_xml_for_slidescore(width, height, tile_size, overlap)

    # Precompute tile bytes for highest resolution level (DeepZoom level 2).
    tiles: dict[tuple[int, int], bytes] = {
        (0, 0): _make_tile((255, 0, 0)),
        (0, 1): _make_tile((0, 255, 0)),
        (1, 0): _make_tile((0, 0, 255)),
        (1, 1): _make_tile((255, 255, 0)),
    }

    async def get_tile_server(request: web.Request) -> web.Response:
        image_id = int(request.query["imageid"])
        payload = {
            "cookiePart": "cookie123",
            "urlPart": f"tiles_{image_id}",
            "expiresOn": "2099-01-01T00:00:00",
        }
        return web.Response(text=json.dumps(payload), content_type="application/json")

    async def get_slide_details(request: web.Request) -> web.Response:
        image_id = int(request.query["imageid"])
        # Study ID is fixed to 42 for this mock.
        payload = {"imageID": image_id, "studyID": 42}
        return web.Response(text=json.dumps(payload), content_type="application/json")

    async def get_image_metadata(request: web.Request) -> web.Response:
        # Mimic the real SlideScore metadata payload as closely as possible.
        payload = {
            "metadata": {
                "Level0TileWidth": 256,
                "Level0TileHeight": 256,
                "OSDTileSize": 256,
                "MppX": 0.25,
                "MppY": 0.25,
                "ObjectivePower": 40.0,
                "BackgroundColor": "#ffffff",
                "LevelCount": 1,
                "Level0Width": 4,
                "Level0Height": 4,
                "Downsamples": [1.0],
                "BoundsX": 0,
                "BoundsY": 0,
                "BoundsWidth": 4,
                "BoundsHeight": 4,
            }
        }
        return web.Response(text=json.dumps(payload), content_type="application/json")

    async def get_dzi(request: web.Request) -> web.Response:
        return web.Response(body=dzi_bytes, content_type="application/xml")

    async def get_tile(request: web.Request) -> web.Response:
        image_id = int(request.match_info["image_id"])
        assert image_id == 1
        level = int(request.match_info["level"])
        tile_name = request.match_info["tile"]
        col, row_with_ext = tile_name.split("_", 1)
        row = row_with_ext.split(".")[0]
        row_idx, col_idx = int(row), int(col)

        if level == 0:
            data = _make_tile((0, 0, 0))
        else:
            data = tiles[(row_idx, col_idx)]
        return web.Response(body=data, content_type="image/jpeg")

    app = web.Application()
    app.add_routes(
        [
            web.get("/Api/GetTileServer", get_tile_server),
            web.get("/Api/GetSlideDetails", get_slide_details),
            web.get("/Api/GetImageMetadata", get_image_metadata),
            web.get("/i/{image_id}/{url_part}/i.dzi", get_dzi),
            web.get("/i/{image_id}/{url_part}/i_files/{level}/{tile}", get_tile),
        ]
    )

    loop = asyncio.new_event_loop()
    runner: Optional[web.AppRunner] = None
    site: Optional[web.TCPSite] = None
    started = threading.Event()

    def _run() -> None:
        nonlocal runner, site
        asyncio.set_event_loop(loop)
        runner_local = web.AppRunner(app)
        loop.run_until_complete(runner_local.setup())
        site_local = web.TCPSite(runner_local, "127.0.0.1", 0)
        loop.run_until_complete(site_local.start())
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


class TestSlideScoreBackend:
    def test_missing_api_token_raises_authentication_error(self) -> None:
        # Ensure environment variable is not affecting this test.
        import os

        os.environ.pop(API_TOKEN_OS_VARIABLE_NAME, None)
        url = "https://mock.slidescore.com/Image/Details?imageId=1&studyId=42"

        with pytest.raises(SlideScoreAuthenticationError):
            _ = SlideScoreSlide(url)

    def test_slidescore_initialisation_and_properties(
        self, slidescore_server: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        assert slide.vendor == "SlideScore"
        assert slide.dimensions == (4, 4)
        assert slide.spacing == (0.25, 0.25)
        assert slide.magnification == 40

        # Tile server properties and cookies should be initialised.
        assert slide.tile_server_properties["cookie_part"] == "cookie123"
        assert "url_part" in slide.tile_server_properties
        assert slide.cookies == {"t": "cookie123"}

        slide.close()

    def test_slidescore_read_region(self, slidescore_server: str) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        region = slide.read_region((0, 0), level=0, size=(4, 4))
        assert isinstance(region, fim.Image)
        width, height, _ = region.dimensions
        assert (width, height) == (4, 4)

        slide.close()

    def test_slidescore_study_id_mismatch_raises_error(
        self,
        slidescore_server: str,
    ) -> None:
        """If the studyId in the URL does not match the server's studyID, a SlideScoreNetworkError is raised."""

        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")
        with pytest.raises(SlideScoreNetworkError):
            slide._validate_study_id({"studyID": 999})
        slide.close()

    def test_fetch_properties_bad_json_raises_network_error(self, slidescore_server: str) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        def fake_fetch(urls: str, **kwargs: Any) -> BytesIO:  # type: ignore[override]
            # Return invalid JSON on metadata endpoint.
            return BytesIO(b"not-json")

        slide.fetch = fake_fetch  # type: ignore[assignment]

        with pytest.raises(SlideScoreNetworkError, match="Failed to parse metadata response"):
            _ = slide._load_slide_metadata()

        slide.close()

    def test_fetch_properties_missing_metadata_key_raises_network_error(self, slidescore_server: str) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        def fake_fetch(urls: str, **kwargs: Any) -> BytesIO:  # type: ignore[override]
            # JSON missing 'metadata' key.
            payload = {"not_metadata": {}}
            return BytesIO(json.dumps(payload).encode("utf-8"))

        slide.fetch = fake_fetch  # type: ignore[assignment]

        with pytest.raises(SlideScoreNetworkError, match="Failed to parse metadata response"):
            _ = slide._load_slide_metadata()

        slide.close()

    def test_fetch_dzi_invalid_xml_raises_network_error(self, slidescore_server: str) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        def fake_fetch(urls: str, **kwargs: Any) -> BytesIO:  # type: ignore[override]
            # Return bytes that are not valid XML.
            return BytesIO(b"<not-xml>")

        slide.fetch = fake_fetch  # type: ignore[assignment]

        with pytest.raises(SlideScoreNetworkError, match="Failed to fetch DZI properties"):
            _ = slide._load_dzi_config()

        slide.close()

    def test_fetch_deepzoom_tile_files_invalid_response_type_raises_network_error(
        self,
        slidescore_server: str,
    ) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        def fake_fetch(urls: list[str], **kwargs: Any) -> BytesIO:  # type: ignore[override]
            # Should be a list of BytesIOs, but we return a single BytesIO to trigger the error.
            return BytesIO(b"tile-bytes")

        slide.fetch = fake_fetch  # type: ignore[assignment]

        with pytest.raises(SlideScoreNetworkError, match="Invalid response type from tile fetch"):
            _ = slide._resolve_deepzoom_tile_paths(level=0, indices=[(0, 0)])

        slide.close()

    def test_fetch_deepzoom_tile_files_wrapped_network_error(self, slidescore_server: str) -> None:
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        slide = SlideScoreSlide(image_url, api_token="dummy-token")

        def fake_fetch(urls: list[str], **kwargs: Any) -> list[BytesIO]:  # type: ignore[override]
            raise SlideScoreNetworkError("inner failure")

        slide.fetch = fake_fetch  # type: ignore[assignment]

        with pytest.raises(SlideScoreNetworkError, match="Failed to fetch deepzoom tiles: inner failure"):
            _ = slide._resolve_deepzoom_tile_paths(level=0, indices=[(0, 0)])

        slide.close()


class TestSlideImageIntegrationWithSlideScore:
    def test_slideimage_read_region_with_slidescore_backend(
        self,
        slidescore_server: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end integration: SlideImage → SlideScore backend → mocked server."""
        image_url = f"{slidescore_server}/Image/Details?imageId=1&studyId=42"
        monkeypatch.setenv(API_TOKEN_OS_VARIABLE_NAME, "dummy-token")

        slide_image = SlideImage.from_file_path(image_url, backend=ImageBackend.SLIDESCORE)

        region = slide_image.read_region((0, 0), scaling=1.0, size=(4, 4))
        assert isinstance(region, fim.Image)
        w, h, _ = region.dimensions
        assert (w, h) == (4, 4)
        assert slide_image.vendor == "SlideScore"

        slide_image.close()
