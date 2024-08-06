# Copyright (c) dlup contributors
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np

cimport numpy as np
from libc.stdint cimport uint8_t, uint64_t


cdef inline int c_floor(float x) noexcept nogil:
    return <int>x - (x < 0 and x != <int>x)


cdef inline int c_ceil(float x) noexcept nogil:
    return <int>x + (x > 0 and x != <int>x)

cdef inline int max_c(int a, int b) noexcept nogil:
    return a if a > b else b

cdef inline int min_c(int a, int b) noexcept nogil:
    return a if (a < b) else b

cdef uint64_t sum_pixels_2d(const uint8_t * data, int width, int height, int stride) noexcept nogil:
    cdef:
        uint64_t sum = 0
        int x, y
        const uint8_t * row_ptr

    for y in range(height):
        row_ptr = data + y * stride
        for x in range(width):
            sum += row_ptr[x]

    return sum


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
        int idx
        float x, y, w, h, mpp
        float image_slide_scaling
        int region_width, region_height
        float scale_factor
        int clipped_w, clipped_h
        int x1, y1, x2, y2
        unsigned char[:, ::1] background_mask_view
        const unsigned char* background_mask_ptr
        const unsigned char* mask_tile_ptr
        Py_ssize_t mask_tile_stride
        long long sum_value
        int error_flag = 0  # 0: No error, 1: MPP zero, 2: Region size zero
        int num_regions = regions_array.shape[0]
        int height = background_mask.shape[0]
        int width = background_mask.shape[1]
        int max_dimension = max_c(width, height)
        int foreground_count = 0

    if not background_mask.flags["C_CONTIGUOUS"]:
        background_mask = np.ascontiguousarray(background_mask)

    num_regions = regions_array.shape[0]
    if foreground_indices.shape[0] < num_regions:
        raise ValueError("foreground_indices array must be at least as long as regions_array")

    background_mask_view = background_mask
    background_mask_ptr = &background_mask_view[0, 0]
    mask_tile_stride = background_mask_view.strides[0]

    with nogil:
        for idx in range(num_regions):
            # num_regions is (x, y, w, h) without constraints
            x, y = regions_array[idx, 0], regions_array[idx, 1]
            w, h = regions_array[idx, 2], regions_array[idx, 3]
            mpp = regions_array[idx, 4]

            if mpp == 0.0:
                error_flag = 1
                break

            image_slide_scaling = image_slide_average_mpp / mpp
            region_width = <int>(image_slide_scaling * image_width)
            region_height = <int>(image_slide_scaling * image_height)

            if region_width == 0 or region_height == 0:
                error_flag = 2
                break

            scale_factor = <float>max_dimension / <float>max_c(region_width, region_height)

            x1 = min_c(width, c_floor(x * scale_factor))
            y1 = min_c(height, c_floor(y * scale_factor))
            x2 = min_c(width, c_ceil((x + w) * scale_factor))
            y2 = min_c(height, c_ceil((y + h) * scale_factor))

            clipped_w = x2 - x1
            clipped_h = y2 - y1

            if x1 >= x2 or y1 >= y2 or clipped_w <= 0 or clipped_h <= 0:
                error_flag = 3
                break

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
    elif error_flag == 3:
        raise RuntimeError(f"Invalid region dimensions (x, y, w, h) = {x, y, w, h}")

    return foreground_count
