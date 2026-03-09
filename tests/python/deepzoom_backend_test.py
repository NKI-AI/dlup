import math
from pathlib import Path

import fim
import numpy as np
import PIL.Image
import pytest

from dlup.backends.deepzoom_backend import DeepZoomSlide, dict_to_snake_case, parse_xml_to_dict


def _write_dzi_xml(path: Path, width: int, height: int, tile_size: int, overlap: int, fmt: str = "png") -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image TileSize="{tile_size}" Overlap="{overlap}" Format="{fmt}"
       xmlns="http://schemas.microsoft.com/deepzoom/2008">
  <Size Width="{width}" Height="{height}"/>
</Image>
"""
    path.write_text(content, encoding="utf-8")


def _make_tile(
    color: tuple[int, int, int],
    size: tuple[int, int],
) -> PIL.Image.Image:
    """Create a solid RGB tile of the given size."""
    w, h = size
    return PIL.Image.fromarray(np.full((h, w, 3), color, dtype=np.uint8), mode="RGB")


def _make_tile_with_overlap(
    tile_color: tuple[int, int, int],
    tile_size: int,
    overlap: int,
    left_color: tuple[int, int, int] | None = None,
    right_color: tuple[int, int, int] | None = None,
    top_color: tuple[int, int, int] | None = None,
    bottom_color: tuple[int, int, int] | None = None,
) -> PIL.Image.Image:
    """Create a DeepZoom tile with proper overlap containing neighboring tile pixels.

    According to Deep Zoom format, tiles are stored with overlap only on inside edges.
    Edge tiles don't have overlap on outside edges, so their stored dimensions vary:
    - Corner tiles: tile_size + overlap (overlap on 2 sides)
    - Edge tiles: tile_size + overlap on one dimension, tile_size + 2*overlap on the other
    - Interior tiles: tile_size + 2*overlap (overlap on all 4 sides)

    Parameters
    ----------
    tile_color : tuple[int, int, int]
        Color for the main tile area (tile_size x tile_size).
    tile_size : int
        Size of the tile (without overlap).
    overlap : int
        Number of overlap pixels on each side.
    left_color, right_color, top_color, bottom_color : tuple[int, int, int] | None
        Colors for overlap regions from neighboring tiles. None means no overlap on that side.

    Returns
    -------
    PIL.Image.Image
        Tile image with dimensions based on which sides have overlap.
    """
    has_left = left_color is not None and overlap > 0
    has_right = right_color is not None and overlap > 0
    has_top = top_color is not None and overlap > 0
    has_bottom = bottom_color is not None and overlap > 0

    width = tile_size + (overlap if has_left else 0) + (overlap if has_right else 0)
    height = tile_size + (overlap if has_top else 0) + (overlap if has_bottom else 0)

    tile_array = np.full((height, width, 3), tile_color, dtype=np.uint8)

    # Fill overlap regions with neighboring tile colors
    if has_left:
        tile_array[:, :overlap] = left_color
    if has_right:
        tile_array[:, -overlap:] = right_color
    if has_top:
        tile_array[:overlap, :] = top_color
    if has_bottom:
        tile_array[-overlap:, :] = bottom_color

    return PIL.Image.fromarray(tile_array, mode="RGB")


@pytest.fixture
def deepzoom_slide(tmp_path: Path) -> DeepZoomSlide:
    """Create a minimal DeepZoom slide on disk with a 4x4 image and 2x2 tiles."""
    dzi_path = tmp_path / "test.dzi"
    width, height = 4, 4
    tile_size, overlap = 2, 0

    _write_dzi_xml(dzi_path, width=width, height=height, tile_size=tile_size, overlap=overlap, fmt="png")

    # Create DeepZoom tile hierarchy: levels 0, 1, 2. We only need tiles at level 0 (for mode)
    # and at the highest-resolution level (2) for read_region.
    tiles_root = tmp_path / "test_files"
    for level in (0, 1, 2):
        (tiles_root / f"{level}").mkdir(parents=True, exist_ok=True)

    # Level 0: single 1x1 logical tile; any small image is fine for mode detection.
    level0_tile = _make_tile((0, 0, 0), (1, 1))
    level0_tile.save(tiles_root / "0" / "0_0.png")

    # Level 2 (highest resolution): 2x2 tiles of size 2x2 covering a 4x4 logical image.
    colors = {
        (0, 0): (255, 0, 0),
        (0, 1): (0, 255, 0),
        (1, 0): (0, 0, 255),
        (1, 1): (255, 255, 0),
    }
    for (row, col), color in colors.items():
        tile = _make_tile(color, (tile_size, tile_size))
        tile.save(tiles_root / "2" / f"{col}_{row}.png")

    return DeepZoomSlide(dzi_path)


class TestDeepZoomSlideBackend:
    def test_dzi_properties_and_geometry(self, deepzoom_slide: DeepZoomSlide) -> None:
        # Properties are loaded from the vips-properties.xml file, which is not created in this test.
        props = deepzoom_slide.properties
        assert isinstance(props, dict) and props == {}

        dzi_props = deepzoom_slide.dzi_properties
        image_props = dzi_props["image"]

        assert image_props["tile_size"] == 2
        assert image_props["overlap"] == 0
        assert image_props["format"] == "png"
        assert image_props["size"]["width"] == 4
        assert image_props["size"]["height"] == 4

        # Level geometry derived from DeepZoom properties
        assert deepzoom_slide.level_count == deepzoom_slide._dz_level_count + 1
        assert deepzoom_slide.level_downsamples == tuple(2**level for level in range(deepzoom_slide.level_count))

        expected_level_dims = []
        for downsample in deepzoom_slide.level_downsamples:
            w = math.ceil(image_props["size"]["width"] / downsample)
            h = math.ceil(image_props["size"]["height"] / downsample)
            expected_level_dims.append((w, h))

        assert deepzoom_slide.level_dimensions == tuple(expected_level_dims)

    def test_mode_property_reads_lowest_level_tile(self, deepzoom_slide: DeepZoomSlide) -> None:
        mode = deepzoom_slide.mode
        assert mode == "RGB"

    @pytest.mark.parametrize(
        "coordinates, size",
        [
            ((0, 0), (4, 4)),  # full image
            ((0, 0), (2, 2)),  # top-left tile
            ((2, 2), (2, 2)),  # bottom-right tile
        ],
    )
    def test_read_region_returns_expected_size(
        self,
        deepzoom_slide: DeepZoomSlide,
        coordinates: tuple[int, int],
        size: tuple[int, int],
    ) -> None:
        region = deepzoom_slide.read_region(coordinates, level=0, size=size)
        assert isinstance(region, fim.Image)
        width, height, _ = region.dimensions
        assert (width, height) == size

    @pytest.mark.parametrize(
        "width,height,tile_size,overlap",
        [
            (4, 4, 2, 0),  # 2x2 grid, no overlap
            (4, 4, 4, 0),  # Single tile, no overlap
            (4, 4, 4, 1),  # Single tile, with overlap (no edge tiles to test)
            (4, 4, 2, 1),  # 2x2 grid, with overlap - tests edge tile overlap handling
            (8, 8, 2, 1),  # 4x4 grid, with overlap - tests larger grid
            (10, 10, 3, 1),  # 4x4 grid (with partial tiles), with overlap
        ],
    )
    def test_read_region_with_varied_tile_size_and_overlap(
        self, tmp_path: Path, width: int, height: int, tile_size: int, overlap: int
    ) -> None:
        """Verify that read_region returns correct size and pixel values for different tile sizes and overlaps.

        We construct a DeepZoom image with distinct colors for each tile and verify that after stitching,
        the final image matches the expected ground truth (simple grid without overlap considerations).
        This ensures that overlap is correctly handled: edge tiles exclude overlap on outside edges,
        and interior tiles correctly crop overlap from all sides.
        """
        dzi_path = tmp_path / f"varied_{width}x{height}_{tile_size}_{overlap}.dzi"
        _write_dzi_xml(dzi_path, width=width, height=height, tile_size=tile_size, overlap=overlap, fmt="png")

        tiles_root = tmp_path / f"varied_{width}x{height}_{tile_size}_{overlap}_files"
        # DeepZoom level index used by backend for level=0 of the slide:
        # level_dz = _level_count - level - 1, with _level_count = ceil(log2(max(width, height))) + 1
        dz_level_count = math.ceil(math.log2(max(width, height)))
        highest_level = dz_level_count

        for level in range(0, highest_level + 1):
            (tiles_root / f"{level}").mkdir(parents=True, exist_ok=True)

        cols = math.ceil(width / tile_size)
        rows = math.ceil(height / tile_size)

        # Generate distinct colors for each tile
        colors: dict[tuple[int, int], tuple[int, int, int]] = {}
        for row in range(rows):
            for col in range(cols):
                r = (row * 50 + col * 30) % 256
                g = (row * 70 + col * 40) % 256
                b = (row * 90 + col * 60) % 256
                colors[(row, col)] = (r, g, b)

        # Create ground truth array
        ground_truth = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                tile_color = colors[(row, col)]
                start_x = col * tile_size
                end_x = min(start_x + tile_size, width)
                start_y = row * tile_size
                end_y = min(start_y + tile_size, height)
                ground_truth[start_y:end_y, start_x:end_x] = tile_color

        # Create tiles with proper DeepZoom overlap containing neighboring tile pixels
        for row in range(rows):
            for col in range(cols):
                tile_color = colors[(row, col)]

                left_color = colors.get((row, col - 1)) if col > 0 else None
                right_color = colors.get((row, col + 1)) if col < cols - 1 else None
                top_color = colors.get((row - 1, col)) if row > 0 else None
                bottom_color = colors.get((row + 1, col)) if row < rows - 1 else None

                tile = _make_tile_with_overlap(
                    tile_color=tile_color,
                    tile_size=tile_size,
                    overlap=overlap,
                    left_color=left_color,
                    right_color=right_color,
                    top_color=top_color,
                    bottom_color=bottom_color,
                )
                tile.save(tiles_root / f"{highest_level}" / f"{col}_{row}.png")

        slide = DeepZoomSlide(dzi_path)
        region = slide.read_region((0, 0), level=0, size=(width, height))
        assert isinstance(region, fim.Image)
        r_w, r_h, _ = region.dimensions
        assert (r_w, r_h) == (width, height)

        region_array = np.array(region.to_numpy())[:, :, :3]
        assert np.array_equal(region_array, ground_truth)


class TestDeepZoomXmlHelpers:
    def test_parse_dzi_roundtrip(self, tmp_path: Path) -> None:
        dzi_path = tmp_path / "roundtrip.dzi"
        _write_dzi_xml(dzi_path, width=512, height=384, tile_size=254, overlap=1, fmt="jpeg")

        parsed = parse_xml_to_dict(dzi_path)
        image_props = parsed["image"]
        assert image_props["tile_size"] == 254
        assert image_props["overlap"] == 1
        assert image_props["format"] == "jpeg"
        assert image_props["size"]["width"] == 512
        assert image_props["size"]["height"] == 384

    def test_dict_to_snake_case_numeric_casting(self) -> None:
        original = {
            "Image": {
                "TileSize": "254",
                "Overlap": "1",
                "Format": "png",
                "Size": {"Width": "1024", "Height": "512"},
            }
        }
        converted = dict_to_snake_case(original)
        image_props = converted["image"]
        assert image_props["tile_size"] == 254
        assert image_props["overlap"] == 1
        assert image_props["format"] == "png"
        assert image_props["size"]["width"] == 1024
        assert image_props["size"]["height"] == 512
