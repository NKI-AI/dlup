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

# pylint: disable=import-error, no-name-in-module
from dlup._background import _get_foreground_indices_numpy
from dlup._exceptions import DlupError
from dlup.annotations import WsiAnnotations

if TYPE_CHECKING:
    from dlup.data.dataset import MaskTypes


def compute_masked_indices(
    slide_image: SlideImage,
    background_mask: MaskTypes,
    regions: collections.abc.Sequence[tuple[float, float, int, int, float]],
    threshold: float | None = 1.0,
) -> npt.NDArray[np.int64]:
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
    npt.NDArray[np.int64]

    """
    if threshold is None:
        return np.arange(0, len(regions), dtype=np.int64)

    if isinstance(background_mask, np.ndarray):
        foreground_indices = np.zeros(len(regions), dtype=np.int64)
        foreground_count = _get_foreground_indices_numpy(
            *slide_image.size,
            slide_image.mpp,
            background_mask,
            np.array(regions, dtype=np.float64),
            threshold,
            foreground_indices,
        )
        masked_indices = foreground_indices[:foreground_count]

    elif isinstance(background_mask, SlideImage):
        slide_image_boolean_mask: npt.NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
        for idx, region in enumerate(regions):
            slide_image_boolean_mask[idx] = _is_foreground_wsiannotations(background_mask, region, threshold)
        masked_indices = np.argwhere(slide_image_boolean_mask).flatten()

    elif isinstance(background_mask, WsiAnnotations):
        wsi_annotations_boolean_mask: npt.NDArray[np.bool_] = np.zeros(len(regions), dtype=bool)
        for idx, region in enumerate(regions):
            wsi_annotations_boolean_mask[idx] = _is_foreground_polygon(slide_image, background_mask, region, threshold)
        masked_indices = np.argwhere(wsi_annotations_boolean_mask).flatten()

    else:
        raise DlupError(f"Unknown background mask type. Got {type(background_mask)}")

    return masked_indices


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
