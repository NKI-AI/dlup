# Copyright (c) dlup contributors
"""
Utilities to handle background / foreground masks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import PIL.Image

from dlup import SlideImage
from dlup._exceptions import DlupError
from dlup.annotations import WsiAnnotations

if TYPE_CHECKING:
    from dlup.data.dataset import MaskTypes


def is_foreground(
    slide_image: SlideImage,
    background_mask: MaskTypes,
    region: tuple[float, float, int, int, float],
    threshold: float | None = 1.0,
) -> bool:
    """Check if a region is foreground.

    Parameters
    ----------
    slide_image : SlideImage
        Slide image to check
    background_mask : np.ndarray or SlideImage or WsiAnnotations
        Background mask to check against
    region : tuple[float, float, int, int, float]
        Region to check
    threshold : float or None
        Threshold to check against. The foreground percentage should be strictly larger than threshold.
        If None anything is foreground. If 1, the region must be completely foreground.
        Other values are in between, for instance if 0.5, the region must be at least 50% foreground.

    Returns
    -------
    bool

    """
    if threshold is None:
        return True

    if isinstance(background_mask, np.ndarray):
        return _is_foreground_numpy(slide_image, background_mask, region, threshold)

    elif isinstance(background_mask, SlideImage):
        return _is_foreground_wsiannotations(background_mask, region, threshold)

    elif isinstance(background_mask, WsiAnnotations):
        return _is_foreground_polygon(slide_image, background_mask, region, threshold)

    else:
        raise DlupError(f"Unknown background mask type. Got {type(background_mask)}")


def _is_foreground_polygon(
    slide_image: SlideImage,
    background_mask: WsiAnnotations,
    region: tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:
    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    scaling = slide_image.get_scaling(mpp)

    polygon_region = background_mask.read_region((x, y), scaling, (w, h))
    total_area = sum([_.area for _ in polygon_region])

    if threshold == 1.0 and total_area == w * h:
        return True

    if total_area / (w * h) > threshold:
        return True

    return False


def _is_foreground_wsiannotations(
    background_mask: SlideImage,
    region: tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:
    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    # TODO: Let us read the mask at its native mpp for speed
    # Can do something as follows, but that is not exposed right now, so that waits for such an implementation.
    # best_level = background_mask._wsi.get_best_level_for_downsample(scaling)
    mask_region_view = background_mask.get_scaled_view(background_mask.get_scaling(mpp))
    mask = mask_region_view.read_region((x, y), (w, h)).convert("L")

    if threshold == 1.0 and np.asarray(mask).mean() == 1:
        return True

    return bool(np.asarray(mask).mean() > threshold)


def _is_foreground_numpy(
    slide_image: SlideImage,
    background_mask: npt.NDArray[np.int_],
    region: tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:
    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    mask_size = np.array(background_mask.shape[:2][::-1])

    region_view = slide_image.get_scaled_view(slide_image.get_scaling(mpp))
    _background_mask = PIL.Image.fromarray(background_mask)

    # Type of background_mask is Any here.
    # The scaling should be computed using the longest edge of the image.
    background_size = (_background_mask.width, _background_mask.height)

    region_size = region_view.size
    max_dimension_index = max(range(len(background_size)), key=background_size.__getitem__)
    scaling = background_size[max_dimension_index] / region_size[max_dimension_index]
    scaled_region = np.array((x, y, w, h)) * scaling
    scaled_coordinates, scaled_sizes = scaled_region[:2], np.ceil(scaled_region[2:]).astype(int)

    mask_tile = np.zeros(scaled_sizes)

    max_boundary = np.tile(mask_size, 2)
    min_boundary = np.zeros_like(max_boundary)
    box = np.clip((*scaled_coordinates, *(scaled_coordinates + scaled_sizes)), min_boundary, max_boundary)
    clipped_w, clipped_h = (box[2:] - box[:2]).astype(int)

    if clipped_h == 0 or clipped_w == 0:
        return False

    mask_tile[:clipped_h, :clipped_w] = np.asarray(
        _background_mask.resize((clipped_w, clipped_h), PIL.Image.BICUBIC, box=box), dtype=float
    )

    if threshold == 1.0 and mask_tile.mean() == 1.0:
        return True

    return bool(mask_tile.mean() > threshold)
