import functools
from dlup.utils.imports import DARWIN_SDK_AVAILABLE
from typing import NamedTuple, Optional, cast
from pathlib import Path

class DarwinV7Metadata(NamedTuple):
    label: str
    color: tuple[int, int, int]
    annotation_type: str

@functools.lru_cache(maxsize=None)
def _get_v7_metadata(filename: Path) -> Optional[dict[tuple[str, str], DarwinV7Metadata]]:
    if not DARWIN_SDK_AVAILABLE:
        raise RuntimeError("`darwin` is not available. Install using `python -m pip install darwin-py`.")
    import darwin.path_utils

    if not filename.is_dir():
        raise RuntimeError("Provide the path to the root folder of the Darwin V7 annotations")

    v7_metadata_fn = filename / ".v7" / "metadata.json"
    if not v7_metadata_fn.exists():
        return None
    v7_metadata = darwin.path_utils.parse_metadata(v7_metadata_fn)
    output = {}
    for sample in v7_metadata["classes"]:
        annotation_type = sample["type"]
        label = sample["name"]
        color = sample["color"][5:-1].split(",")
        if color[-1] != "1.0":
            raise RuntimeError("Expected A-channel of color to be 1.0")
        rgb_colors = (int(color[0]), int(color[1]), int(color[2]))

        output[(label, annotation_type)] = DarwinV7Metadata(
            label=label, color=rgb_colors, annotation_type=annotation_type
        )
    return output


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    if "#" not in hex_color:
        if hex_color == "black":
            return 0, 0, 0
    hex_color = hex_color.lstrip("#")

    # Convert the string from hex to an integer and extract each color component
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def _get_geojson_color(properties: dict[str, str | list[int]]) -> Optional[tuple[int, int, int]]:
    """Parse the properties dictionary of a GeoJSON object to get the color.

    Arguments
    ---------
    properties : dict
        The properties dictionary of a GeoJSON object.

    Returns
    -------
    Optional[tuple[int, int, int]]
        The color of the object as a tuple of RGB values.
    """
    color = properties.get("color", None)
    if color is None:
        return None

    return cast(tuple[int, int, int], tuple(color))


def _get_geojson_z_index(properties: dict[str, str | list[int]]) -> Optional[int]:
    """Parse the properties dictionary of a GeoJSON object to get the z_index`.

    Arguments
    ---------
    properties : dict
        The properties dictionary of a GeoJSON object.

    Returns
    -------
    Optional[tuple[int, int, int]]
        The color of the object as a tuple of RGB values.
    """
    z_index = properties.get("z_index", None)
    if z_index is None:
        return None

    return cast(int, z_index)
