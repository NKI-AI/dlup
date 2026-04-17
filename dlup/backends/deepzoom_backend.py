from __future__ import annotations

import abc
import itertools
import math
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import fim
import numpy as np
import PIL

from dlup._types import PathLike
from dlup.backends.common import AbstractSlideBackend

RELEVANT_VIPS_PROPERTIES = {
    "openslide.vendor": str,
    "openslide.mpp-x": float,
    "openslide.mpp-y": float,
    "openslide.objective-power": int,
    "openslide.bounds-height": int,
    "openslide.bounds-width": int,
    "openslide.bounds-x": int,
    "openslide.bounds-y": int,
    "openslide.quickhash-1": str,
    "vips-loader": str,
    "bands": int,
}

_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")

TileResponseTypes = Union[list[str], list[BytesIO]]


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


class AbstractDeepZoomSlide(AbstractSlideBackend):
    """
    Abstract base class for deepzoom experimental_backends
    """

    def __init__(self, filename: PathLike, **kwargs: Any):
        """
        Parameters
        ----------
        filename : PathLike
            Path to image (or URL for remote backends).
        **kwargs : Any
            Additional arguments passed to other parent classes.
        """
        super().__init__(filename, **kwargs)
        self._properties = self._load_slide_metadata()
        self._dzi_properties = self._load_dzi_config()

        if self.properties.get("mpp_x") is not None and self.properties.get("mpp_y") is not None:
            self._spacings = [(float(self.properties["mpp_x"]), float(self.properties["mpp_y"]))]

        self._dz_level_count = math.ceil(
            math.log2(
                max(
                    self.dzi_properties["image"]["size"]["width"],
                    self.dzi_properties["image"]["size"]["height"],
                )
            )
        )
        self._tile_size = (self.dzi_properties["image"]["tile_size"],) * 2
        self._overlap = self.dzi_properties["image"]["overlap"]

        self._level_count = self._dz_level_count + 1
        self._downsamples = [2**level for level in range(self._level_count)]
        self._shapes = [
            (
                math.ceil(self.dzi_properties["image"]["size"]["width"] / downsample),
                math.ceil(self.dzi_properties["image"]["size"]["height"] / downsample),
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
        return self._properties

    @property
    def dzi_properties(self) -> dict[str, Any]:
        """Deep Zoom Image (DZI) properties of slide."""
        return self._dzi_properties

    @abc.abstractmethod
    def _load_slide_metadata(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def _load_dzi_config(self) -> dict[str, Any]: ...

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

        This is determined by the mode of the DeepZoom tile at level 0. This is an image of size 1x1 that should
        exist for every DeepZoom image.
        """
        if not hasattr(self, "_mode"):
            _region = PIL.Image.open(self._resolve_deepzoom_tile_paths(0, [(0, 0)])[0])
            self._mode: str = _region.mode  # type: ignore
        return self._mode

    @property
    def slide_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Returns the bounds of the slide. These can be smaller than the image itself."""
        if self.properties.get("bounds_x") is None or self.properties.get("bounds_y") is None:
            return (0, 0), self.dimensions

        bounds_offset = (self.properties["bounds_x"], self.properties["bounds_y"])
        bounds_size = (self.properties["bounds_width"], self.properties["bounds_height"])
        return bounds_offset, bounds_size

    @property
    @abc.abstractmethod
    def tile_files_root(self) -> str:
        """Returns the path where deep zoom tiles are stored as a string."""
        ...

    def _resolve_deepzoom_tile_paths(self, level: int, indices: list[tuple[int, int]]) -> TileResponseTypes:
        """Resolve paths or ByteIO objects for tile indices of a DeepZoom level.

        These paths/objects will be opened with Pillow and stitched together in ``read_region``.

        Parameters
        ----------
        level : int
            Deep zoom level for tiles
        indices : list[tuple[int, int]]
            List of (row, col) tuples for column and row at specified deepzoom level

        Returns
        -------
        list[str | BytesIO]
            List of file paths or ByteIO objects for unprocessed DeepZoom tiles.
        """
        tile_files_root = self.tile_files_root
        file_format = self.dzi_properties["image"]["format"]
        return [f"{tile_files_root}/{level}/{col}_{row}.{file_format}" for row, col in indices]

    def read_region(self, coordinates: tuple[Any, ...], level: int, size: tuple[int, int]) -> fim.Image:
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
        fim.Image
            The requested region.
        """
        x, y = coordinates
        width, height = size
        tile_w, tile_h = self._tile_size
        overlap = self._overlap
        max_col, max_row = self._num_cols_rows[level]

        # Calculate the range of rows and columns for tiles covering the specified region
        start_row = y // tile_h
        end_row = min(math.ceil((y + height) / tile_h), max_row)
        start_col = x // tile_w
        end_col = min(math.ceil((x + width) / tile_w), max_col)

        tile_indices = list(itertools.product(range(start_row, end_row), range(start_col, end_col)))
        level_dz = self._level_count - level - 1
        tile_paths = self._resolve_deepzoom_tile_paths(level_dz, tile_indices)

        region = fim.Image.black(width=width, height=height)
        for (row, col), tile_path in zip(tile_indices, tile_paths):
            # TODO: Use fim.Image.from_file instead of PIL.Image.open for better performance.
            # There is no from_jpeg method, so we need to use PIL.Image.open and convert to numpy.
            tile_image = fim.Image.from_numpy(np.array(PIL.Image.open(tile_path)))

            tile_x = col * tile_w
            tile_y = row * tile_h

            # Calculate intersection of tile with requested region
            region_start_x = max(0, tile_x - x)
            region_end_x = min(width, tile_x - x + tile_w)
            region_start_y = max(0, tile_y - y)
            region_end_y = min(height, tile_y - y + tile_h)

            # Calculate crop region within the tile
            crop_start_x = region_start_x - (tile_x - x)
            crop_end_x = region_end_x - (tile_x - x)
            crop_start_y = region_start_y - (tile_y - y)
            crop_end_y = region_end_y - (tile_y - y)

            # Adjust crop coordinates to exclude overlap on edges (only needed when overlap > 0)
            if overlap > 0:
                # Edge tiles don't have overlap on outside edges, so we only crop overlap from interior edges
                # Check if this tile is on an edge by comparing tile column/row indices to the grid boundaries
                is_left_edge = col == 0
                is_right_edge = col == max_col - 1
                is_top_edge = row == 0
                is_bottom_edge = row == max_row - 1

                # Crop overlap from left side (unless this is the leftmost tile)
                # Leftmost tiles don't have left overlap in the file, so we don't need to skip it
                if not is_left_edge:
                    crop_start_x += overlap
                    crop_end_x += overlap

                # Crop overlap from top side (unless this is the topmost tile)
                if not is_top_edge:
                    crop_start_y += overlap
                    crop_end_y += overlap

                # Clamp crop coordinates to actual tile file dimensions as a safety check
                actual_tile_width = (
                    tile_w + (overlap if not is_left_edge else 0) + (overlap if not is_right_edge else 0)
                )
                actual_tile_height = (
                    tile_h + (overlap if not is_top_edge else 0) + (overlap if not is_bottom_edge else 0)
                )
                crop_end_x = min(crop_end_x, actual_tile_width)
                crop_end_y = min(crop_end_y, actual_tile_height)
            else:
                # No overlap: tiles are exactly tile_w x tile_h, so we clamp to tile dimensions
                crop_end_x = min(crop_end_x, tile_w)
                crop_end_y = min(crop_end_y, tile_h)

            # Crop and paste the tile region
            crop_width = max(0, crop_end_x - crop_start_x)
            crop_height = max(0, crop_end_y - crop_start_y)
            if crop_width > 0 and crop_height > 0:
                cropped_tile = tile_image.crop(position=(crop_start_x, crop_start_y), size=(crop_width, crop_height))  # type: ignore
                region = region.paste(cropped_tile, x=region_start_x, y=region_start_y)

        return region


class DeepZoomSlide(AbstractDeepZoomSlide):
    def __init__(self, filename: str | Path):
        super().__init__(filename)

    def _load_slide_metadata(self) -> dict[str, Any]:
        """Fetch properties of the slide. A `vips-properties.xml` file will be generated by vips when extracting
        the pyramid. Correctness not tested for vips-loader other than `openslideload`
        """
        vips_properties_file = Path(self.tile_files_root) / "vips-properties.xml"
        if not vips_properties_file.exists():
            return {}
        # Don't convert to snake case for now to keep original vips-property names
        vips_properties = parse_xml_to_dict(vips_properties_file, _to_snake_case=False)["image"]["properties"]
        relevant_properties = {
            relevant_key.split("openslide.")[-1]: cast_fn(vips_properties[relevant_key])
            for relevant_key, cast_fn in RELEVANT_VIPS_PROPERTIES.items()
            if relevant_key in vips_properties
        }
        if relevant_properties.get("vips-loader", "") != "openslideload":
            raise NotImplementedError(
                f"Properties not implemented for vips-loader {relevant_properties.get('vips-loader')}."
            )
        # Convert to snake case naming convention in the end
        return dict_to_snake_case(relevant_properties)

    def _load_dzi_config(self) -> dict[str, Any]:
        """Fetch DeepZoom properties from .dzi file. Cast every property, except for `Format`, to integers."""
        return parse_xml_to_dict(self._filename)

    @property
    def tile_files_root(self) -> str:
        """Returns path where deep zoom tiles are stored. Default is folder named `<file_name>_files` at the same
        location where .dzi file is stored."""
        return str(Path(self._filename).parent / f"{Path(self._filename).stem}_files")

    def close(self) -> None:
        """Close the underlying slide"""
        pass


# Utility functions for parsing vips-properties.xml and .dzi files
def parse_xml_to_dict(file_path: PathLike | BytesIO, _to_snake_case: bool = True) -> dict[str, Any]:
    """Parse XML file (vips-properties.xml or .dzi) into a dictionary. `vips-properties.xml` files will extract
    every property name-value pair in `properties` key.

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
    namespace = "".join(root.tag.partition("}")[:2]) if "}" in root.tag else ""
    root_tag = root.tag.replace(namespace, "")
    parsed_dict: dict[str, dict[str, Any]] = {root_tag: dict(root.attrib)}
    for elem in root:
        tag = elem.tag.replace(namespace, "")
        attributes = extract_vips_properties(elem, namespace=namespace) if tag == "properties" else dict(elem.attrib)
        parsed_dict[root_tag][tag] = attributes
    return dict_to_snake_case(parsed_dict) if _to_snake_case else parsed_dict


def extract_vips_properties(properties_elem: ET.Element, namespace: str) -> dict[str, Any]:
    """
    Extract 'properties' section from vips-properties.xml, with name-value pairs.

    Parameters
    ----------
    properties_elem : xml.etree.ElementTree.Element
        The 'properties' XML element.
    namespace : str
        The namespace for the XML document.

    Returns
    -------
    dict[str, Any]
        Dictionary of properties with name-value pairs.
    """
    properties: dict[str, Any] = {}
    for prop in properties_elem.findall(f".//{namespace}property"):
        name_elem = prop.find(f"{namespace}name")
        if name_elem is None or name_elem.text is None:
            continue

        name = name_elem.text.strip()
        if not name:
            continue

        value_elem = prop.find(f"{namespace}value")
        value_text = value_elem.text.strip() if value_elem is not None and value_elem.text is not None else None

        properties[name] = value_text

    return properties


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
    return_dict: dict[str, Any] = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            # Recursively convert dictionary keys
            v = dict_to_snake_case(v)
        elif isinstance(v, str):
            # Cast to float, int or leave as string
            if _NUMERIC_RE.match(v):
                v = float(v) if "." in v else int(v)

        # Convert key to snake_case (i.e. no dashes/spaces, lowercase and underscore before capital letters)
        if isinstance(k, str):
            k = re.sub("([a-z0-9])([A-Z])", r"\1_\2", re.sub("(.)([A-Z][a-z]+)", r"\1_\2", k)).lower().replace("-", "_")
        return_dict[k] = v
    return return_dict
