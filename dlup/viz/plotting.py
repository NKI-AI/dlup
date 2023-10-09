# Copyright (c) dlup Contributors
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw

from dlup.annotations import Point, Polygon


def plot_2d(
    image: PIL.Image.Image,
    mask: npt.NDArray[np.int_] | None = None,
    mask_colors=None,
    mask_alpha: int = 70,
    geometries=None,
    geometries_color_map=None,
) -> PIL.Image.Image:
    """
    Plotting utility to overlay masks and geometries (Points, Polygons) on top of the image.

    Parameters
    ----------
    image : PIL.Image
    mask : np.ndarray
        Integer array
    mask_colors : dict
        A dictionary mapping the integer value of the mask to a PIL color value.
    mask_alpha : int
        Value between 0-100 defining the transparency of the overlays
    geometries : list
        List of Point or Polygon
    geometries_color_map : dict
        Dictionary mapping label names to the PIL color value

    Returns
    -------
    PIL.Image
    """
    image = image.convert("RGBA")

    if mask is not None:
        # Get unique values
        unique_vals = sorted(list(np.unique(mask)))
        for idx in unique_vals:
            if idx == 0:
                continue
            color = PIL.ImageColor.getcolor(mask_colors[idx], "RGBA")
            curr_mask = PIL.Image.fromarray(((mask == idx)[..., np.newaxis] * color).astype(np.uint8), mode="RGBA")
            alpha_channel = PIL.Image.fromarray(
                ((mask == idx) * int(mask_alpha * 255 / 100)).astype(np.uint8), mode="L"
            )
            curr_mask.putalpha(alpha_channel)
            image = PIL.Image.alpha_composite(image.copy(), curr_mask.copy()).copy()

    if geometries is not None:
        draw = PIL.ImageDraw.Draw(image)
        for data in geometries:
            if isinstance(data, Point):
                r = 10
                x, y = data.coords[0]
                _points = [(x - r, y - r), (x + r, y + r)]
                draw.ellipse(_points, outline=geometries_color_map[data.label], width=3)

            elif isinstance(data, Polygon):
                coordinates = data.exterior.coords
                draw.polygon(
                    coordinates,
                    fill=None,
                    outline=geometries_color_map[data.label],
                    width=3,
                )

            else:
                raise RuntimeError(f"Type {type(data)} not implemented.")

    return image.convert("RGB")
