// Copyright 2024 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

inline int c_floor(float x) noexcept {
  return static_cast<int>(x) - (x < 0 && x != static_cast<int>(x));
}

inline int c_ceil(float x) noexcept {
  return static_cast<int>(x) + (x > 0 && x != static_cast<int>(x));
}

inline int max_c(int a, int b) noexcept {
  return std::max(a, b);
}

inline int min_c(int a, int b) noexcept {
  return std::min(a, b);
}

uint64_t sum_pixels_2d(const uint8_t* data, int width, int height,
                       int stride) noexcept {
  uint64_t sum = 0;
  for (int y = 0; y < height; ++y) {
    const uint8_t* row_ptr = data + y * stride;
    for (int x = 0; x < width; ++x) {
      sum += row_ptr[x];
    }
  }
  return sum;
}

int get_foreground_indices_numpy(int image_width, int image_height,
                                 float image_slide_average_mpp,
                                 py::array_t<uint8_t> background_mask,
                                 py::array_t<double> regions_array,
                                 float threshold,
                                 py::array_t<int64_t> foreground_indices) {
  auto background_mask_info = background_mask.request();
  auto regions_array_info = regions_array.request();
  auto foreground_indices_info = foreground_indices.request();

  const uint8_t* background_mask_ptr =
      static_cast<const uint8_t*>(background_mask_info.ptr);
  auto background_mask_shape = background_mask_info.shape;
  auto background_mask_strides = background_mask_info.strides;

  const double* regions_ptr =
      static_cast<const double*>(regions_array_info.ptr);
  auto regions_shape = regions_array_info.shape;

  int64_t* foreground_indices_ptr =
      static_cast<int64_t*>(foreground_indices_info.ptr);

  int num_regions = regions_shape[0];
  int height = background_mask_shape[0];
  int width = background_mask_shape[1];
  int max_dimension = max_c(width, height);

  int foreground_count = 0;
  int error_flag = 0;

  for (int idx = 0; idx < num_regions; ++idx) {
    float x = regions_ptr[idx * 5 + 0];
    float y = regions_ptr[idx * 5 + 1];
    float w = regions_ptr[idx * 5 + 2];
    float h = regions_ptr[idx * 5 + 3];
    float mpp = regions_ptr[idx * 5 + 4];

    if (mpp == 0.0f) {
      error_flag = 1;
      break;
    }

    float image_slide_scaling = image_slide_average_mpp / mpp;
    int region_width = static_cast<int>(image_slide_scaling * image_width);
    int region_height = static_cast<int>(image_slide_scaling * image_height);

    if (region_width == 0 || region_height == 0) {
      error_flag = 2;
      break;
    }

    float scale_factor =
        static_cast<float>(max_dimension) / max_c(region_width, region_height);

    int x1 = min_c(width, c_floor(x * scale_factor));
    int y1 = min_c(height, c_floor(y * scale_factor));
    int x2 = min_c(width, c_ceil((x + w) * scale_factor));
    int y2 = min_c(height, c_ceil((y + h) * scale_factor));

    int clipped_w = x2 - x1;
    int clipped_h = y2 - y1;

    if (x1 >= x2 || y1 >= y2 || clipped_w <= 0 || clipped_h <= 0) {
      error_flag = 3;
      break;
    }

    const uint8_t* mask_tile_ptr =
        background_mask_ptr + y1 * background_mask_strides[0] + x1;
    uint64_t sum_value = sum_pixels_2d(mask_tile_ptr, clipped_w, clipped_h,
                                       background_mask_strides[0]);

    if (sum_value > threshold * clipped_w * clipped_h) {
      foreground_indices_ptr[foreground_count++] = idx;
    }
  }

  if (error_flag == 1) {
    throw std::invalid_argument("mpp cannot be zero");
  } else if (error_flag == 2) {
    throw std::runtime_error("region_width or region_height cannot be zero");
  } else if (error_flag == 3) {
    throw std::runtime_error("Invalid region dimensions");
  }

  return foreground_count;
}

PYBIND11_MODULE(_background, m) {
  m.doc() = "Foreground indices computation module";

  m.def("get_foreground_indices_numpy", &get_foreground_indices_numpy,
        py::arg("image_width"), py::arg("image_height"),
        py::arg("image_slide_average_mpp"), py::arg("background_mask"),
        py::arg("regions_array"), py::arg("threshold"),
        py::arg("foreground_indices"),
        "Compute foreground indices given background mask and regions array.");
}
