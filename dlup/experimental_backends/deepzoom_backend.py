from __future__ import annotations

import functools
import itertools
import math
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any, Union

from PIL import Image

from dlup.backends.common import AbstractSlideBackend
from dlup.types import PathLike

METADATA_CACHE = 128
RELEVANT_VIPS_PROPERTIES = {
    "openslide.vendor": str,
    "openslide.mpp-x": float,
    "openslide.mpp-y": float,
    "openslide.objective-power": int,
    "openslide.bounds-height": int,
    "openslide.bounds-width": int,
    "openslide.bounds-x": int,
    "openslide.bounds-y": int,
    "vips-loader": str,
}

TileResponseTypes = Union[str, BytesIO]


def parse_xml_to_dict(file_path: PathLike | BytesIO, _to_snake_case: bool = True) -> dict[str, Any]:
    """Parse XML file with name space. vips-properties.xml files will extract every property name-value pair in
    `properties`.

    Parameters
    ----------
    file_path : Pathlike or BytesIO
        Path or BytesIO object to XML file
    _to_snake_case : bool, optional
        Convert keys to snake case naming convention, by default True

    Returns
    -------
    dict[str, Any]
        Parsed XML file as a dictionary. Name space will be replaced with an empty string.
    """
    root = ET.parse(file_path).getroot()
    namespace = root.tag.split("}")[0] + "}" if len(root.tag.split("}")) > 1 else ""
    root_tag = root.tag.replace(namespace, "")
    parsed_dict: dict[str, dict[str, Any]] = {root_tag: dict(root.attrib)}
    for elem in root:
        tag = elem.tag.replace(namespace, "")
        if tag == "properties":
            properties = {}
            for prop in elem.findall(f".//{namespace}property"):
                name = prop.find(f"{namespace}name")
                if name is None:
                    continue
                value = prop.find(f"{namespace}value")
                properties[str(name.text)] = value.text if value is not None else value
            parsed_dict["properties"] = properties
        else:
            parsed_dict[root_tag][tag] = dict(elem.attrib)
    return dict_to_snake_case(parsed_dict) if _to_snake_case else parsed_dict


def dict_to_snake_case(dictionary: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert all keys in a dictionary to snake case naming convention. String values will be
    converted to floats and integers if appropriate.

    Parameters
    ----------
    dictionary : dict[str, Any]
        Dictionary with keys using Camel/Pascal naming convention.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys using Snake case naming convention and values as strings, floats and integers.
    """
    return_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            # Recursively convert dictionary keys
            v = dict_to_snake_case(v)
        elif isinstance(v, str):
            # Cast to float, int or leave as string
            if re.compile(r"^\d+(\.\d+)?$").match(v):
                v = float(v) if "." in v else int(v)

        # Convert key to snake_case (i.e. no dashes/spaces, lowercase and underscore before capital letters)
        if isinstance(k, str):
            k = (
                re.sub("([a-z0-9])([A-Z])", r"\1_\2", re.sub("(.)([A-Z][a-z]+)", r"\1_\2", k))
                .lower()
                .replace("-", "_")
            )
        return_dict[k] = v
    return return_dict


def open_slide(filename: PathLike) -> "DeepZoomSlide":
    """
    Read slide with DeepZoomSlide backend. The input file should be a <slide_name>.dzi file with the deep zoom tiles
    in a folder <slide_name>_files

    Parameters
    ----------
    filename : PathLike
        DZI file for slide.
    """
    return DeepZoomSlide(filename)


class DeepZoomSlide(AbstractSlideBackend):

    def __init__(self, filename: PathLike):
        super().__init__(filename)
        if self.properties.get("mpp_x") is not None and self.properties.get("mpp_y") is not None:
            self._spacings = [(float(self.properties["mpp_x"]), float(self.properties["mpp_y"]))]

        self._dz_level_count = math.ceil(
            math.log2(
                max(
                    self.dz_properties["image"]["size"]["width"],
                    self.dz_properties["image"]["size"]["height"],
                )
            )
        )
        self._tile_size = (self.dz_properties["image"]["tile_size"],) * 2
        self._overlap = self.dz_properties["image"]["overlap"]

        self._level_count = self._dz_level_count + 1
        self._downsamples = [2**level for level in range(self._level_count)]
        self._shapes = [
            (
                math.ceil(self.dz_properties["image"]["size"]["width"] / downsample),
                math.ceil(self.dz_properties["image"]["size"]["height"] / downsample),
            )
            for downsample in self._downsamples
        ]

        self._num_cols_rows = [
            (
                width // self._tile_size[0] + int((width % self._tile_size[0]) > 0),
                height // self._tile_size[1] + int((height % self._tile_size[1]) > 0),
            )
            for width, height in self._shapes
        ]

    @property
    def properties(self) -> dict[str, Any]:
        """Properties of slide"""
        if not hasattr(self, "_properties"):
            self._properties = self._fetch_properties()
        return self._properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_properties(self) -> dict[str, Any]:
        """Fetch properties of the slide. The `vips-properties.xml` file will be generated by vips when extracting
        the pyramid. Correctness not tested for vips-loader other than `openslideload`
        """
        vips_properties_file = Path(self.tile_files) / "vips-properties.xml"
        if not vips_properties_file.exists():
            return {}
        # Don't convert to snake case for now to keep original vips-property names
        vips_properties_dict = parse_xml_to_dict(vips_properties_file, _to_snake_case=False)
        properties = {
            relevant_key.split("openslide.")[-1]: cast_fn(vips_properties_dict["properties"][relevant_key])
            for relevant_key, cast_fn in RELEVANT_VIPS_PROPERTIES.items()
            if relevant_key in vips_properties_dict["properties"]
        }
        if properties.get("vips-loader") is None or properties.get("vips-loader") != "openslideload":
            raise NotImplementedError(f"Properties not implemented for vips-loader {properties.get('vips-loader')}.")
        # Convert to snake case naming convention in the end
        return dict_to_snake_case(properties)

    @property
    def dz_properties(self) -> dict[str, Any]:
        """DeepZoom properties of slide"""
        if not hasattr(self, "_dz_properties"):
            self._dz_properties = self._fetch_dz_properties()
        return self._dz_properties

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_dz_properties(self) -> dict[str, Any]:
        """Fetch DeepZoom properties from .dzi file. Cast every property, except for `Format`, to integers."""
        return parse_xml_to_dict(self._filename)

    @property
    def magnification(self) -> float | None:
        """Returns the objective power at which the WSI was sampled."""
        value = self.properties.get("objective_power")
        if value is not None:
            return int(value)
        return value

    @property
    def vendor(self) -> str | None:
        """Returns the scanner vendor."""
        return self.properties.get("vendor")

    @property
    def mode(self) -> str:
        """Returns the mode of the deep zoom tiles.
        NOTE: When generating deepzoom pyramid with VIPS, this could be CYMK and differ from the original slide
        """
        if not hasattr(self, "_mode"):
            self._mode = self._fetch_mode()
        return self._mode

    @functools.lru_cache(maxsize=METADATA_CACHE)
    def _fetch_mode(self) -> str:
        """Returns the mode of the deepzoom tile at level 0. This is an image of size 1x1 that should exist."""
        return str(Image.open(self.retrieve_deepzoom_tiles(0, [(0, 0)])[0]).mode)

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide. These can be smaller than the image itself."""
        if self.properties.get("bounds_x") is None or self.properties.get("bounds_y") is None:
            return (0, 0), self.dimensions

        # If MRXS file is generate with --angle d90, x and width should be switched with y and height respectively
        bounds_offset = (self.properties["bounds_x"], self.properties["bounds_y"])
        bounds_size = (self.properties["bounds_width"], self.properties["bounds_height"])
        return bounds_offset, bounds_size

    @property
    def tile_files(self) -> PathLike:
        """Returns path where deep zoom tiles are stored. Default is folder named `<file_name>_files` at the same
        location where .dzi file is stored."""
        return Path(self._filename).parent / f"{Path(self._filename).stem}_files"

    def retrieve_deepzoom_tiles(self, level: int, indices: list[tuple[int, int]]) -> list[TileResponseTypes]:
        """Retrieve paths or ByteIO objects for tile indices of deepzoom level. These tiles will be opened with Pillow
        and stitched together in `read_region`

        Parameters
        ----------
        level : int
            Deep zoom level for tiles
        indices : list[tuple[int, int]]
            List of (col, row) tuples for column and row at specified deepzoom level

        Returns
        -------
        list[Path | BytesIO]
            List of file paths or ByteIO objects for unprocessed DeepZoom tiles.
        """
        tile_files_root = self.tile_files
        file_format = self.dz_properties["image"]["format"]
        return [f"{tile_files_root}/{level}/{col}_{row}.{file_format}" for col, row in indices]

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
        level_end_col, level_end_row = self._num_cols_rows[level]
        tile_w, tile_h = self._tile_size

        x, y = (coordinates[0] // level_downsample, coordinates[1] // level_downsample)
        w, h = size

        # Calculate the range of rows and columns for tiles covering the specified region
        start_row = y // tile_h
        end_row = min(math.ceil((y + h) / tile_h), level_end_row)
        start_col = x // tile_w
        end_col = min(math.ceil((x + w) / tile_w), level_end_col)

        indices = list(itertools.product(range(start_col, end_col), range(start_row, end_row)))
        level_dz = self._level_count - level - 1
        tile_files = self.retrieve_deepzoom_tiles(level_dz, indices)

        _region = Image.new(self.mode, size, (255,) * len(self.mode))
        for (col, row), tile_file in zip(indices, tile_files):
            _region_tile = Image.open(tile_file)
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

            # All but edge tiles have overlap pixels outside of tile
            if col > 0:
                crop_start_x += self._overlap
                crop_end_x += self._overlap
            if col == level_end_col:
                crop_end_x -= self._overlap

            if row > 0:
                crop_start_y += self._overlap
                crop_end_y += self._overlap
            if row == level_end_row:
                crop_end_y -= self._overlap

            _cropped_region_tile = _region_tile.crop((crop_start_x, crop_start_y, crop_end_x, crop_end_y))
            _region.paste(_cropped_region_tile, (img_start_x, img_start_y))
        return _region

    def close(self) -> None:
        """Close the underlying slide"""
        return
