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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

inline int c_floor(double x) noexcept {
  return static_cast<int>(x) - (x < 0 && x != static_cast<int>(x));
}

inline int c_ceil(double x) noexcept {
  return static_cast<int>(x) + (x > 0 && x != static_cast<int>(x));
}

inline int min_c(int a, int b) noexcept {
  return std::min(a, b);
}

uint64_t sum_pixels_2d(const uint8_t* data, int width, int height,
                       int64_t stride) noexcept {
  uint64_t sum = 0;
  for (int y = 0; y < height; ++y) {
    const uint8_t* row_ptr = data + y * stride;
    for (int x = 0; x < width; ++x) {
      sum += row_ptr[x];
    }
  }
  return sum;
}

// All ndarray arguments are zero-copy views (no `nb::numpy` tag, no copy):
// they reference the caller's buffer directly.
int get_foreground_indices_numpy(
    int image_width, int image_height, double image_slide_average_mpp,
    nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>
        background_mask,
    nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>
        regions_array,
    double threshold,
    nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>
        foreground_indices) {
  const uint8_t* background_mask_ptr = background_mask.data();
  const double* regions_ptr = regions_array.data();
  int64_t* foreground_indices_ptr = foreground_indices.data();

  const int num_regions = static_cast<int>(regions_array.shape(0));
  const int height = static_cast<int>(background_mask.shape(0));
  const int width = static_cast<int>(background_mask.shape(1));
  const int64_t row_stride = background_mask.stride(0);

  int foreground_count = 0;

  // Shared context appended to every error so a caller can immediately see the
  // slide/mask geometry that produced the failure.
  auto geometry_context = [&]() -> std::string {
    return " [slide_size=(" + std::to_string(image_width) + ", " +
           std::to_string(image_height) +
           "), slide_mpp=" + std::to_string(image_slide_average_mpp) +
           ", mask_shape=(" + std::to_string(height) + ", " +
           std::to_string(width) +
           "), num_regions=" + std::to_string(num_regions) + "]";
  };

  for (int idx = 0; idx < num_regions; ++idx) {
    double x = regions_ptr[idx * 5 + 0];
    double y = regions_ptr[idx * 5 + 1];
    double w = regions_ptr[idx * 5 + 2];
    double h = regions_ptr[idx * 5 + 3];
    double mpp = regions_ptr[idx * 5 + 4];

    auto region_context = [&]() -> std::string {
      return " (region[" + std::to_string(idx) + "] x=" + std::to_string(x) +
             ", y=" + std::to_string(y) + ", w=" + std::to_string(w) +
             ", h=" + std::to_string(h) + ", mpp=" + std::to_string(mpp) + ")";
    };

    if (mpp == 0.0) {
      throw std::invalid_argument("Region mpp cannot be zero." +
                                  region_context() + geometry_context());
    }

    double image_slide_scaling = image_slide_average_mpp / mpp;
    int region_width = static_cast<int>(image_slide_scaling * image_width);
    int region_height = static_cast<int>(image_slide_scaling * image_height);

    if (region_width == 0 || region_height == 0) {
      throw std::runtime_error(
          "Region has zero extent after scaling the slide to the region mpp "
          "(scaled_slide_size=(" +
          std::to_string(region_width) + ", " + std::to_string(region_height) +
          "), scaling=" + std::to_string(image_slide_scaling) +
          "). This usually means the slide mpp or size is inconsistent with "
          "the "
          "requested tiling mpp." +
          region_context() + geometry_context());
    }

    // Map region coordinates (in slide pixels at the region mpp) onto the mask
    // grid. The mask and the scaled slide are rounded to integers
    // independently, so their aspect ratios differ slightly; using a single
    // scale factor for both axes lets the minor axis overshoot by ~1 pixel at
    // the slide edge. Scale each axis by its own ratio to keep edge tiles in
    // bounds.
    double scale_x = static_cast<double>(width) / region_width;
    double scale_y = static_cast<double>(height) / region_height;

    int x1 = min_c(width, c_floor(x * scale_x));
    int y1 = min_c(height, c_floor(y * scale_y));
    int x2 = min_c(width, c_ceil((x + w) * scale_x));
    int y2 = min_c(height, c_ceil((y + h) * scale_y));

    int clipped_w = x2 - x1;
    int clipped_h = y2 - y1;

    if (x1 >= x2 || y1 >= y2 || clipped_w <= 0 || clipped_h <= 0) {
      throw std::runtime_error(
          "Region projects onto an empty area of the mask. The scaled region "
          "box collapsed after clipping to the mask bounds "
          "(mask_box=[x1=" +
          std::to_string(x1) + ", y1=" + std::to_string(y1) +
          ", x2=" + std::to_string(x2) + ", y2=" + std::to_string(y2) +
          "], clipped_size=(" + std::to_string(clipped_w) + ", " +
          std::to_string(clipped_h) + "), scale_x=" + std::to_string(scale_x) +
          ", scale_y=" + std::to_string(scale_y) + ", scaled_slide_size=(" +
          std::to_string(region_width) + ", " + std::to_string(region_height) +
          ")). This typically happens when the slide size/mpp reported by the "
          "backend does not match the mask that was generated for this slide, "
          "so the region falls outside the mask." +
          region_context() + geometry_context());
    }

    const uint8_t* mask_tile_ptr = background_mask_ptr + y1 * row_stride + x1;
    uint64_t sum_value =
        sum_pixels_2d(mask_tile_ptr, clipped_w, clipped_h, row_stride);

    if (sum_value > threshold * clipped_w * clipped_h) {
      foreground_indices_ptr[foreground_count++] = idx;
    }
  }

  return foreground_count;
}

NB_MODULE(_background, m) {
  m.doc() = "Foreground indices computation module";

  m.def("get_foreground_indices_numpy", &get_foreground_indices_numpy,
        nb::arg("image_width"), nb::arg("image_height"),
        nb::arg("image_slide_average_mpp"), nb::arg("background_mask"),
        nb::arg("regions_array"), nb::arg("threshold"),
        nb::arg("foreground_indices"),
        "Compute foreground indices given background mask and regions array.");
}
