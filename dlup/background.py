# Copyright (c) dlup contributors
"""
Utilities to handle background / foreground masks.
"""
from __future__ import annotations

import collections
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from dlup import SlideImage
from dlup._background import _is_foreground_numpy  # pylint: disable=no-name-in-module
from dlup._exceptions import DlupError
from dlup.annotations import WsiAnnotations

if TYPE_CHECKING:
    from dlup.data.dataset import MaskTypes


def is_foreground(
    slide_image: SlideImage,
    background_mask: MaskTypes,
    regions: collections.abc.Sequence[tuple[float, float, int, int, float]],
    threshold: float | None = 1.0,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_]]:
    """Filter the regions to foreground data. This can be either an `np.ndarray` or `SlideImage` or `WsiAnnotations`.
    Using numpy arrays is the fastest way to filter the background.

    Parameters
    ----------
    slide_image : SlideImage
        Slide image to check
    background_mask : np.ndarray or SlideImage or WsiAnnotations
        Background mask to check against
    regions : collections.abc.Sequence[tuple[float, float, int, int, float]]
        Regions to check
    threshold : float or None
        Threshold to check against. The foreground percentage should be strictly larger than threshold.
        If None anything is foreground. If 1, the region must be completely foreground.
        Other values are in between, for instance if 0.5, the region must be at least 50% foreground.

    Returns
    -------
    npt.NDArray[np.bool_], npt.NDArray[np.int_]

    """
    if threshold is None:
        return np.ones(len(regions), dtype=bool), np.arange(0, len(regions))

    boolean_mask: npt.NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
    if isinstance(background_mask, np.ndarray):
        _is_foreground_numpy(slide_image, background_mask, list(regions), boolean_mask, threshold)
        masked_indices = np.argwhere(boolean_mask).flatten()

    elif isinstance(background_mask, SlideImage):
        for idx, region in enumerate(regions):
            boolean_mask[idx] = _is_foreground_wsiannotations(background_mask, region, threshold)
        masked_indices = np.argwhere(boolean_mask).flatten()

    elif isinstance(background_mask, WsiAnnotations):
        for idx, region in enumerate(regions):
            boolean_mask[idx] = _is_foreground_polygon(slide_image, background_mask, region, threshold)
        masked_indices = np.argwhere(boolean_mask).flatten()

    else:
        raise DlupError(f"Unknown background mask type. Got {type(background_mask)}")

    return boolean_mask, masked_indices


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
    total_area = sum(_.area for _ in polygon_region)

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

    # We need to make sure the w, h are fitting inside the image and otherwise clip to that
    mask_size = mask_region_view.size
    # Now we must make sure that (x, y) + (w, h) <= mask_size
    w, h = min(w, mask_size[0] - int(x)), min(h, mask_size[1] - int(y))

    # If there is a color_map it then pyvips will read it as a RGBA array.
    # In that case we can just average over the last access and convert it to a boolean value
    # We need to drop the A channel, as it is not relevant for the mask.
    mask = mask_region_view.read_region((x, y), (w, h)).colourspace("srgb").numpy()[..., :3]

    mask = np.sum(mask, axis=-1)
    mask = (mask > 0).astype(int)

    if threshold == 1.0 and np.asarray(mask).mean() == 1:
        return True

    return bool(np.asarray(mask).mean() > threshold)


# def _is_foreground_numpy(
#     slide_image: SlideImage,
#     background_mask: npt.NDArray[np.int_],
#     regions,
#     threshold: float = 1.0,
# ) -> bool:
#
#     boolean_mask: npt.NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
#     for idx, region in enumerate(regions):
#
#         x, y, w, h, mpp = region
#
#         mask_size = np.array(background_mask.shape[:2][::-1])
#         region_view = slide_image.get_scaled_view(slide_image.get_scaling(mpp))
#         region_size = region_view.size
#
#         max_dimension_index = np.argmax(mask_size)
#         scaling = mask_size[max_dimension_index] / region_size[max_dimension_index]
#         scaled_region = np.array((x, y, w, h)) * scaling
#         scaled_coordinates, scaled_sizes = scaled_region[:2], np.ceil(scaled_region[2:]).astype(int)
#
#         max_boundary = np.tile(mask_size, 2)
#         min_boundary = np.zeros_like(max_boundary)
#         box = np.clip((*scaled_coordinates, *(scaled_coordinates + scaled_sizes)), min_boundary, max_boundary)
#         clipped_w, clipped_h = (box[2:] - box[:2]).astype(int)
#
#         if clipped_h == 0 or clipped_w == 0:
#             return False
#
#         x1, y1, x2, y2 = box.astype(int)
#         mask_tile = background_mask[y1:y2, x1:x2]
#
#         if mask_tile.shape != (clipped_h, clipped_w):
#             zoom_factors = (clipped_h / mask_tile.shape[0], clipped_w / mask_tile.shape[1])
#             mask_tile = ndimage.zoom(mask_tile, zoom_factors, order=1)
#
#         if threshold == 1.0 and mask_tile.mean() == 1.0:
#             return True
#
#         boolean_mask[idx] = bool(mask_tile.mean() > threshold)
#
#     return boolean_mask
