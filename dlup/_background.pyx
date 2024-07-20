# Copyright (c) dlup contributors
# cython: language_level=3
import cython
import numpy as np

cimport numpy as np
from libc.math cimport ceil, floor

ctypedef fused SlideImage:
    object

@cython.cdivision(True)
cdef inline float safe_divide(float a, float b) nogil:
    return a / b if b != 0 else 0

cdef inline int max_c(int a, int b) nogil:
    return a if a > b else b

cdef inline int min_c(int a, int b) nogil:
    return a if a < b else b

import cython

from libc.stdint cimport uint8_t, uint64_t

import cython

from libc.stdint cimport uint8_t, uint64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t sum_pixels_2d(const uint8_t * data, int width, int height, int stride) nogil:
    cdef:
        uint64_t sum = 0
        int x, y
        int aligned_width
        const uint8_t * row_ptr
        uint64_t row_sum

    if width >= 8:
        aligned_width = width - (width % 8)

        for y in range(height):
            row_ptr = data + y * stride
            row_sum = 0

            # Process 8 pixels at a time
            for x in range(0, aligned_width, 8):
                row_sum += (
                    row_ptr[x] +
                    row_ptr[x + 1] +
                    row_ptr[x + 2] +
                    row_ptr[x + 3] +
                    row_ptr[x + 4] +
                    row_ptr[x + 5] +
                    row_ptr[x + 6] +
                    row_ptr[x + 7]
                )

            # Handle remaining pixels in the row
            for x in range(aligned_width, width):
                row_sum += row_ptr[x]

            sum += row_sum
    else:
        # Handle case where width < 8
        for y in range(height):
            row_ptr = data + y * stride
            row_sum = 0
            for x in range(width):
                row_sum += row_ptr[x]
            sum += row_sum

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
def _get_foreground_indices_numpy(
    int image_width,
    int image_height,
    float image_slide_average_mpp,
    np.ndarray background_mask,
    np.ndarray[np.float64_t, ndim=2] regions_array,
    float threshold,
    np.ndarray[np.int64_t, ndim=1] foreground_indices,
):
    cdef:
        int idx, num_regions
        float x, y, w, h, mpp
        int width, height, max_dimension
        float image_slide_scaling
        int region_width, region_height
        float scale_factor
        float scaled_region[4]
        int clipped_w, clipped_h
        int x1, y1, x2, y2
        unsigned char[:, ::1] background_mask_view
        const unsigned char* background_mask_ptr
        const unsigned char* mask_tile_ptr
        Py_ssize_t mask_tile_stride
        long long sum_value
        int error_flag = 0  # 0: No error, 1: MPP zero, 2: Region size zero
        int foreground_count = 0

    num_regions = regions_array.shape[0]
    if foreground_indices.shape[0] < num_regions:
        raise ValueError("foreground_indices array must be at least as long as regions_array")

    background_mask_view = background_mask
    background_mask_ptr = &background_mask_view[0, 0]
    mask_tile_stride = background_mask_view.strides[0]

    height, width = background_mask.shape[:2]
    max_dimension = max_c(width, height)

    for idx in range(num_regions):
        x, y = regions_array[idx, 0], regions_array[idx, 1]
        w, h = regions_array[idx, 2], regions_array[idx, 3]
        mpp = regions_array[idx, 4]

        if mpp == 0:
            error_flag = 1
            break

        image_slide_scaling = safe_divide(image_slide_average_mpp, mpp)
        region_width = <int>(image_slide_scaling * image_width)
        region_height = <int>(image_slide_scaling * image_height)

        if region_width == 0 or region_height == 0:
            error_flag = 2
            break

        scale_factor = safe_divide(max_dimension, <float>max_c(region_width, region_height))

        x1 = min_c(width, <int> floor(x * scale_factor))
        y1 = min_c(height, <int> floor(y * scale_factor))
        x2 = min_c(width, <int> ceil((x + w) * scale_factor))
        y2 = min_c(height, <int> ceil((y + h) * scale_factor))

        clipped_w = x2 - x1
        clipped_h = y2 - y1

        if clipped_h == 0 or clipped_w == 0:
            continue

        mask_tile_ptr = background_mask_ptr + y1 * mask_tile_stride + x1
        sum_value = sum_pixels_2d(mask_tile_ptr, clipped_w, clipped_h, mask_tile_stride)

        if sum_value > threshold * clipped_w * clipped_h:
            foreground_indices[foreground_count] = idx
            foreground_count += 1

    if error_flag == 1:
        raise ValueError("mpp cannot be zero")
    elif error_flag == 2:
        raise RuntimeError("region_width or region_height cannot be zero")

    return foreground_count
