# coding=utf-8
# Copyright (c) dlup contributors

"""Algorithms for background segmentation.

This module contains a collection of background extraction algorithms specifically
tailored for histopathology.

Currently implemented:

-   FESI (Foreground Extraction for Histopathological Whole-Slide Imaging).
-   Improved FESI.

Check their respective documentations for references.
"""

from enum import Enum
from functools import partial
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import PIL.Image
import scipy.ndimage as ndi
import skimage.filters
import skimage.morphology
import skimage.segmentation

import dlup
import dlup.tiling
from dlup import SlideImage
from dlup._exceptions import DlupError
from dlup.annotations import WsiAnnotations

_GenericIntArray = Union[np.ndarray, Iterable[int]]


def _is_close(_seeds, _start) -> bool:
    """
    Helper function for the FESI algorithms.

    Parameters
    ----------
    _seeds : list
    _start : list

    Returns
    -------
    bool
    """
    _start = np.asarray(_start)
    max_dist = 500
    for seed in _seeds:
        _seed = np.asarray(seed[0])
        if np.sqrt(((_start - _seed) ** 2).sum()) <= max_dist:
            return True
    return False


def _fesi_common(image: np.ndarray) -> np.ndarray:
    """
    Common functionality for FESI and Improved FESI.

    Parameters
    ----------
    image : np.ndarray
        2D gray-scale image.

    Returns
    -------
    np.ndarray
        Tissue mask
    """
    blurred = skimage.filters.gaussian(image, sigma=1, truncate=5)
    mask = ((blurred >= np.mean(blurred)) * 150).astype("uint8")
    inverse_mask = ((blurred < np.mean(blurred)) * 255).astype("uint8")

    dseed = ndi.distance_transform_edt(inverse_mask)
    mask = skimage.filters.median(mask, np.ones((7, 7))) + 100

    kernel = np.ones((5, 5), np.uint8)
    # TODO: Binary opening is faster
    mask = skimage.morphology.opening(mask, footprint=kernel)
    max_point = np.unravel_index(np.argmax(dseed, axis=None), dseed.shape)

    skimage.segmentation.flood_fill(mask, seed_point=max_point, new_value=0, in_place=True)
    mask[mask > 0] = 255
    distance = ndi.distance_transform_edt(mask)

    final_mask = mask.copy()
    maximal_distance = distance.max()
    global_max = distance.max()
    seeds: List = []
    while maximal_distance > 0:
        start = np.unravel_index(distance.argmax(), distance.shape)
        if (maximal_distance > 0.6 * global_max) or _is_close(seeds, start[::-1]):
            skimage.segmentation.flood_fill(final_mask, seed_point=start, new_value=200, in_place=True)
            seeds.append((start, maximal_distance))
        skimage.segmentation.flood_fill(mask, seed_point=start, new_value=0, in_place=True)
        distance[mask == 0] = 0
        maximal_distance = distance.max()

    final_mask[final_mask != 200] = 0
    final_mask[final_mask == 200] = 255
    return final_mask.astype(bool)


def improved_fesi(image: np.ndarray) -> np.ndarray:
    """Combination of original and improved FESI algorithms.

    Extract foreground from background from H&E WSIs combining the original
    and improved FESI algorithms.
    This is an implementation of the Improved FESI algorithm as described in [1]
    with the additional multiplication of the original FESI algorithm as described in [2].
    In particular, they change the color space of the input image from BGR
    to LAB and the value of the first two channels, the lightness and the red/green value
    are mapped to the maximum intensity value. This image is subsequently changed to gray-scale
    and binarized using the mean value of the gray-scale image as threshold.
    It is this image which is passed to the Gaussian filter, rather than the absolute value
    of the Laplacian as in the original FESI algorithm.

    Parameters
    ----------
    image : np.ndarray
        2D RGB image, with color channels at the last axis.

    Returns
    -------
    np.ndarray
        Tissue mask

    References
    ----------
    .. [1] https://arxiv.org/pdf/2006.06531.pdf
    .. [2] https://www.lfb.rwth-aachen.de/bibtexupload/pdf/BUG15fesi.pdf
    """
    chnl = [0, 1]
    img_bgr = image[..., [2, 1, 0]]
    img_hsv = skimage.color.rgb2hsv(img_bgr)

    lab_bgr = skimage.color.rgb2lab(img_bgr)
    lab_bgr[..., chnl] = 100

    gray_bgr = skimage.color.rgb2gray(skimage.color.lab2rgb(lab_bgr))
    tissue_bgr = np.abs(skimage.filters.laplace(skimage.color.rgb2gray(img_bgr), ksize=3))
    gray_bgr = (gray_bgr >= np.mean(gray_bgr)) * 1.0

    tissue_rgb_hsv_1 = tissue_bgr * img_hsv[..., 1] * gray_bgr

    return _fesi_common(tissue_rgb_hsv_1)


def fesi(image: np.ndarray) -> np.ndarray:
    """
    Extract foreground from background from H&E WSIs using the FESI algorithm [1].

    Parameters
    ----------
    image : np.ndarray
        2D RGB image, with color channels at the last axis.

    Returns
    -------
    np.ndarray
        Tissue mask

    References
    ----------
    .. [1] https://www.lfb.rwth-aachen.de/bibtexupload/pdf/BUG15fesi.pdf
    """
    gray = skimage.color.rgb2gray(image)
    gray = np.abs(skimage.filters.laplace(gray, ksize=3))
    return _fesi_common(gray)


def next_power_of_2(x):
    """Returns the smallest greater than x, power of 2 value.

    https://stackoverflow.com/a/14267557/576363
    """
    x = int(x)
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def get_mask(slide: dlup.SlideImage, mask_func: Callable = improved_fesi, minimal_size: int = 512) -> np.ndarray:
    """
    Compute a tissue mask for a Slide object.

    The mask will be computed on a thumbnail of the original slide
    of the closest power of 2 of a resolution of 10 microns per pixel,
    or minimal_size, whatever is largest.
    https://github.com/alexander-rakhlin/he_stained_fg_extraction

    Parameters
    ----------
    slide : dlup.slide.Slide
    mask_func : callable
        Function mapping 2D RGB image with channels on the last axis.
    minimal_size : int
        Minimal thumbnail size.

    Returns
    -------
    np.ndarray
        Tissue mask of thumbnail
    """
    max_slide = max(slide.size)
    size = max_slide * slide.mpp / 10  # Size is 10 mpp
    # Max sure it is at least max_slide and a power of 2.
    # TODO: maybe this power is not needed
    size = int(max([next_power_of_2(size), min([minimal_size, max_slide])]))

    # TODO: max should be determined by system memory
    mask = mask_func(np.asarray(slide.get_thumbnail(size=(size, size))))
    return mask.astype(np.uint8)


def is_foreground(
    slide_image: SlideImage,
    background_mask: Union[np.ndarray, SlideImage, WsiAnnotations],
    region: Tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:

    if isinstance(background_mask, np.ndarray):
        return is_foreground_numpy(slide_image, background_mask, region, threshold)

    elif isinstance(background_mask, SlideImage):
        return is_foreground_wsiannotations(background_mask, region, threshold)

    elif isinstance(background_mask, WsiAnnotations):
        return is_foreground_polygon(slide_image, background_mask, region, threshold)

    else:
        raise DlupError(f"Unknown background mask type. Got {type(background_mask)}")


def is_foreground_polygon(
    slide_image: SlideImage,
    background_mask: WsiAnnotations,
    region: Tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:

    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    scaling = slide_image.get_scaling(mpp)

    polygon_region = background_mask.read_region((x, y), scaling, (w, h))
    total_area = sum([_.area for _ in polygon_region])

    if total_area / (w * h) > threshold:
        return True

    return False


def is_foreground_wsiannotations(
    background_mask: SlideImage,
    region: Tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:

    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    # TODO: Let us read the mask at its native mpp for speed
    # Can do something as follows, but that is not exposed right now, so that waits for such an implementation.
    # best_level = background_mask._wsi.get_best_level_for_downsample(scaling)
    mask_region_view = background_mask.get_scaled_view(background_mask.get_scaling(mpp))
    mask = mask_region_view.read_region((x, y), (w, h)).convert("L")

    return np.asarray(mask).mean() > threshold


def is_foreground_numpy(
    slide_image: SlideImage,
    background_mask: np.ndarray,
    region: Tuple[float, float, int, int, float],
    threshold: float = 1.0,
) -> bool:

    # Let's get the region view from the slide image.
    x, y, w, h, mpp = region

    mask_size = np.array(background_mask.shape[:2][::-1])

    region_view = slide_image.get_scaled_view(slide_image.get_scaling(mpp))
    background_mask = PIL.Image.fromarray(background_mask)

    # Type of background_mask is Any here.
    # The scaling should be computed using the longest edge of the image.
    background_size = (background_mask.width, background_mask.height)  # type: ignore

    region_size = region_view.size
    max_dimension_index = max(range(len(background_size)), key=background_size.__getitem__)
    scaling = background_size[max_dimension_index] / region_size[max_dimension_index]
    scaled_region = np.array((x, y, w, h)) * scaling
    scaled_coordinates, scaled_sizes = scaled_region[:2], np.ceil(scaled_region[2:]).astype(int)

    mask_tile = np.zeros(scaled_sizes)

    max_boundary = np.tile(mask_size, 2)
    min_boundary = np.zeros_like(max_boundary)
    box = np.clip((*scaled_coordinates, *(scaled_coordinates + scaled_sizes)), min_boundary, max_boundary)  # type: ignore
    clipped_w, clipped_h = (box[2:] - box[:2]).astype(int)

    if clipped_h == 0 or clipped_w == 0:
        return False

    mask_tile[:clipped_h, :clipped_w] = np.asarray(
        background_mask.resize((clipped_w, clipped_h), PIL.Image.BICUBIC, box=box), dtype=float  # type: ignore
    )
    return mask_tile.mean() > threshold


class AvailableMaskFunctions(Enum):
    """
    Class that maps strngs to the currently implemented mask functions as defined in dlup/background.py

    The method returns the function from a string input, e.g.

    >>> mask_func = AvailableMaskFunctions('fesi')

    after which `mask_func` can be passed to dlup.background.get_mask

    If you want to get a list of the strings that can be given that map to functions:

    >>> valid_inputs = [key for key in AvailableMaskFunctions.__members__]
    """

    fesi = partial(fesi)
    improved_fesi = partial(improved_fesi)

    def __call__(self, *args):
        return self.value(*args)
