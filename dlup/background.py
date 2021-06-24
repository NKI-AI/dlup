# coding=utf-8
# Copyright (c) dlup contributors

"""Algorithms for background segmentation.

This module contains a collection of background extraction algorithms specifically
tailored for histopathology.
"""

from typing import Callable, List

import numpy as np
import PIL.Image
import scipy.ndimage as ndi
import skimage.filters
import skimage.morphology
import skimage.segmentation

from dlup import SlideImage, SlideImageTiledRegionView


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
    mask = skimage.morphology.opening(mask, selem=kernel)
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


def get_mask(slide: SlideImage, mask_func: Callable = improved_fesi, minimal_size: int = 512) -> np.ndarray:
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
    mask = mask_func(slide.get_thumbnail(size=(size, size)))
    return mask.astype(np.uint8)


def foreground_tiles_coordinates_mask(
    background_mask: np.ndarray, tiled_region_view: SlideImageTiledRegionView, threshold: float = 1.0
):
    """Generate a numpy boolean mask that can be applied to tiles coordinates.

    A tiled region view contains the tiles coordinates as a flattened grid.
    This function returns an array of boolean values being True if
    the tile is considered foreground and False otherwise.


    Parameters
    ----------
    background_mask :
        Binary mask representing of the background generated with get_mask().
    tiled_region_view :
        Target tiled_region_view we want to generate the mask for.
    threshold :
        Threshold of amount of foreground required to classify a tile as foreground.

    Returns
    -------
    np.ndarray:
        Boolean array of the same shape as the tiled_region_view.coordinates.
    """
    slide_image_region_view = tiled_region_view.region_view
    mask_size = np.array(background_mask.shape[:2][::-1])

    background_mask = PIL.Image.fromarray(background_mask)

    # Type of background_mask is Any here.
    scaling = background_mask.width / slide_image_region_view.size[0]  # type: ignore
    scaled_tile_size = np.array(tiled_region_view.tile_size) * scaling
    scaled_tile_size = scaled_tile_size.astype(int)
    scaled_coordinates = tiled_region_view.coordinates * scaling

    # Generate an array of boxes.
    boxes = np.hstack([scaled_coordinates, scaled_coordinates + scaled_tile_size])

    # Let's clip values outside boundaries.
    max_a = np.tile(mask_size, 2)
    min_a = np.zeros_like(max_a)
    boxes = np.clip(boxes, min_a, max_a)  # type: ignore

    # Fill in the mask with boolean values if the mean number of pixels
    # of tissue surpasses a threshold.
    mask = np.empty(len(boxes), dtype=bool)
    for i, b in enumerate(boxes):
        mask_tile = background_mask.resize(scaled_tile_size, PIL.Image.BICUBIC, box=b)  # type: ignore
        mask_tile = np.asarray(mask_tile, dtype=float)
        mask[i] = mask_tile.mean() >= threshold

    return mask
