# Copyright (c) dlup contributors
# cython: language_level=3
import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, floor
from libc.stdlib cimport free, malloc

ctypedef fused SlideImage:
    object

@cython.cdivision(True)
cdef inline float safe_divide(float a, float b) nogil:
    return a / b if b != 0 else 0

cdef inline int max_c(int a, int b) nogil:
    return a if a > b else b

cdef inline int min_c(int a, int b) nogil:
    return a if a < b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void resize_mask_nearest_uint8(const unsigned char* input_mask, int input_width, int input_height, int input_stride,
                                    unsigned char* output_mask, int output_width, int output_height) nogil:
    cdef:
        int error_flag = 0 # 0: No error, 1: Input size zero
        float x_ratio
        float y_ratio
        int x, y, src_x, src_y

    x_ratio = safe_divide(<float>input_width, output_width)
    y_ratio = safe_divide(<float>input_height, output_height)

    for y in range(output_height):
        for x in range(output_width):
            src_x = <int>(x * x_ratio)
            src_y = <int>(y * y_ratio)
            output_mask[y * output_width + x] = input_mask[src_y * input_stride + src_x]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned long long sum_pixels(const unsigned char* data, int size, int stride) nogil:
    cdef:
        unsigned long long sum = 0
        int i, j
        int aligned_size = size - (size % 8)

    if size < 8:
        for i in range(size):
            sum += data[i]
        return sum

    for i in range(0, aligned_size, 8):
        sum += (
            data[i] +
            data[i + 1] +
            data[i + 2] +
            data[i + 3] +
            data[i + 4] +
            data[i + 5] +
            data[i + 6] +
            data[i + 7]
        )

    # Handle remaining pixels
    for i in range(aligned_size, size):
        sum += data[i]

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
def _is_foreground_numpy(
    SlideImage slide_image,
    np.ndarray background_mask,
    np.ndarray[np.float32_t, ndim=2] regions_array,
    np.ndarray[np.uint8_t, ndim=1] boolean_mask,
    float threshold
):
    cdef:
        int idx, num_regions
        float x, y, w, h, mpp
        int width, height, max_dimension
        float image_slide_scaling, image_slide_average_mpp
        int region_width, region_height, image_width, image_height
        float scaling
        float scaled_region[4]
        int box[4]
        int clipped_w, clipped_h
        int x1, y1, x2, y2
        float mean_value
        int i
        unsigned char[:, ::1] background_mask_view
        const unsigned char* background_mask_ptr
        const unsigned char* mask_tile_ptr
        Py_ssize_t mask_tile_stride
        unsigned char* temp_mask
        long long sum_value
        int total_pixels
        int ix, iy
        int error_flag = 0  # 0: No error, 1: MPP zero, 2: Region size zero

    num_regions = regions_array.shape[0]
    if boolean_mask.shape[0] != num_regions:
        raise ValueError("boolean_mask must have the same length as regions")

    background_mask_view = background_mask
    background_mask_ptr = &background_mask_view[0, 0]
    mask_tile_stride = background_mask_view.strides[0]

    height, width = background_mask.shape[:2]
    max_dimension = max_c(width, height)

    image_slide_average_mpp = slide_image.mpp
    image_width, image_height = slide_image.size

    temp_mask = <unsigned char*>malloc(max_dimension * max_dimension * sizeof(unsigned char))
    if not temp_mask:
        raise MemoryError("Failed to allocate temporary buffer")

    try:
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

            scaling = safe_divide(max_dimension, <float>max_c(region_width, region_height))
            scaled_region[0] = x * scaling
            scaled_region[1] = y * scaling
            scaled_region[2] = w * scaling
            scaled_region[3] = h * scaling

            box[0] = max_c(0, min_c(width, <int>floor(scaled_region[0])))
            box[1] = max_c(0, min_c(height, <int>floor(scaled_region[1])))
            box[2] = max_c(0, min_c(width, <int>ceil(scaled_region[0] + scaled_region[2])))
            box[3] = max_c(0, min_c(height, <int>ceil(scaled_region[1] + scaled_region[3])))

            clipped_w = box[2] - box[0]
            clipped_h = box[3] - box[1]

            if clipped_h == 0 or clipped_w == 0:
                continue

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            mask_tile_ptr = background_mask_ptr + y1 * mask_tile_stride + x1

            if (y2 - y1) != clipped_h or (x2 - x1) != clipped_w:
                resize_mask_nearest_uint8(mask_tile_ptr, x2 - x1, y2 - y1, mask_tile_stride,
                                          temp_mask, clipped_w, clipped_h)
                sum_value = sum_pixels(temp_mask, clipped_w * clipped_h, clipped_w)
            else:
                sum_value = 0
                for iy in range(clipped_h):
                    sum_value += sum_pixels(mask_tile_ptr + iy * mask_tile_stride, clipped_w, 1)

            total_pixels = clipped_w * clipped_h
            mean_value = safe_divide(<float>sum_value, total_pixels)

            if mean_value > threshold:
                boolean_mask[idx] = 1

    finally:
        free(temp_mask)

    if error_flag == 1:
        raise ValueError("mpp cannot be zero")
    elif error_flag == 2:
        raise RuntimeError("region_width or region_height cannot be zero")
