# Copyright (c) dlup contributors
"""Utilities to handle backends."""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any, Callable

from dlup._types import PathLike
from dlup.utils.imports import AIOHTTP_AVAILABLE


def parse_xml_to_dict(file_path: PathLike | io.BytesIO, _to_snake_case: bool = True) -> dict[str, Any]:
    """Parse XML file (DeepZoom DZI or vips-properties.xml) into a dictionary. `vips-properties.xml` files will extract
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
    properties = {}
    for prop in properties_elem.findall(f".//{namespace}property"):
        name_elem = prop.find(f"{namespace}name")
        if name_elem is None:
            continue
        value_elem = prop.find(f"{namespace}value")
        properties[str(name_elem.text)] = value_elem.text if value_elem is not None else None

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
            k = re.sub("([a-z0-9])([A-Z])", r"\1_\2", re.sub("(.)([A-Z][a-z]+)", r"\1_\2", k)).lower().replace("-", "_")
        return_dict[k] = v
    return return_dict


class ImageBackend(Enum):
    """Available image experimental_backends."""

    from dlup.backends.deepzoom_backend import DeepZoomSlide
    from dlup.backends.openslide_backend import OpenSlideSlide
    from dlup.backends.pyvips_backend import PyVipsSlide
    from dlup.backends.tifffile_backend import TifffileSlide

    if AIOHTTP_AVAILABLE:
        from dlup.backends.slidescore_backend import SlideScoreSlide

    OPENSLIDE: Callable[[PathLike], OpenSlideSlide] = OpenSlideSlide
    PYVIPS: Callable[[PathLike], PyVipsSlide] = PyVipsSlide
    TIFFFILE: Callable[[PathLike], TifffileSlide] = TifffileSlide
    DEEPZOOM: Callable[[PathLike], DeepZoomSlide] = DeepZoomSlide
    if AIOHTTP_AVAILABLE:
        SLIDESCORE: Callable[[PathLike], SlideScoreSlide] = SlideScoreSlide

    def __call__(self, *args: "ImageBackend" | str) -> Any:
        return self.value(*args)
