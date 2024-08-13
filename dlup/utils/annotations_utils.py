from typing import Optional, cast


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
