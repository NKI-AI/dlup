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
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "aifocore/math/ndarray.h"
#include "aifocore/math/python/ndarray_numpy.h"
#include "dlup/geometry/marching_squares.h"
#include "dlup/geometry/polygon.h"

namespace py = pybind11;

namespace {

/**
 * @brief Helper template to process arrays of different types.
 *
 * Converts input array to double and calls FindContours.
 *
 * @tparam T Input array element type.
 * @param image Input binary image as a numpy array.
 * @param level The iso-value level at which to extract contours.
 * @return Vector of Polygon objects from the marching squares algorithm.
 */
template <typename T>
std::vector<std::shared_ptr<dlup::geometry::Polygon>> ProcessArray(
    const py::array_t<T>& image, double level) {
  py::buffer_info buf_info = image.request();

  if (buf_info.ndim != 2) {
    throw std::invalid_argument("Input array must be 2-dimensional");
  }

  const std::size_t height = static_cast<std::size_t>(buf_info.shape[0]);
  const std::size_t width = static_cast<std::size_t>(buf_info.shape[1]);

  // For types that are already double, use zero-copy view
  if constexpr (std::is_same_v<T, double>) {
    aifocore::math::NDArrayView<double, 2> ndarray_view(
        static_cast<double*>(buf_info.ptr), {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  } else {
    // For other types, convert to double
    // This is still efficient for small binary masks
    std::vector<double> converted_data(height * width);
    const T* src = static_cast<T*>(buf_info.ptr);

    for (std::size_t i = 0; i < height * width; ++i) {
      converted_data[i] = static_cast<double>(src[i]);
    }

    aifocore::math::NDArrayView<double, 2> ndarray_view(converted_data.data(),
                                                        {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  }
}

}  // namespace

/**
 * @brief Python wrapper for FindContours that accepts numpy arrays.
 *
 * Accepts common numeric types: uint8, int32, float32, float64.
 *
 * @param image Input binary image as a numpy array.
 * @param level The iso-value level at which to extract contours.
 * @return List of Polygon objects representing the contours.
 */
std::vector<std::shared_ptr<dlup::geometry::Polygon>> FindContoursPython(
    const py::array& image, double level) {
  // Dispatch based on input array type
  if (py::isinstance<py::array_t<uint8_t>>(image)) {
    return ProcessArray(image.cast<py::array_t<uint8_t>>(), level);
  } else if (py::isinstance<py::array_t<int32_t>>(image)) {
    return ProcessArray(image.cast<py::array_t<int32_t>>(), level);
  } else if (py::isinstance<py::array_t<float>>(image)) {
    return ProcessArray(image.cast<py::array_t<float>>(), level);
  } else if (py::isinstance<py::array_t<double>>(image)) {
    return ProcessArray(image.cast<py::array_t<double>>(), level);
  } else {
    throw std::invalid_argument(
        "Input array must be of type uint8, int32, float32, or float64");
  }
}

PYBIND11_MODULE(marching_squares, m) {
  m.doc() = "Marching squares algorithm for contour extraction";

  m.def("find_contours", &FindContoursPython, py::arg("image"),
        py::arg("level") = 0.5,
        R"pbdoc(
        Find iso-valued contours in a binary 2D array using marching squares.

        Uses the marching squares algorithm to compute contours at the specified
        level. Array values are linearly interpolated to provide better precision.
        Contours that touch the image border are automatically closed by adding
        corner points along the border.

        Contours wind counter-clockwise around low-valued regions. For binary
        masks (0/1), uint8 is the most efficient dtype. NaN values in float
        arrays are skipped. Uses low-value connectivity for ambiguous cases.

        Args:
            image (ndarray): Input binary image of shape (M, N) in which to find
                contours. Must be a 2D numpy array. Accepts uint8, int32, float32,
                or float64 types. For binary masks, uint8 is recommended.
            level (float, optional): The iso-value level at which to extract contours.
                Default is 0.5. For binary masks, use 0.5 for centered contours, or
                values like 0.1 or 0.9 for pixel-aligned boundaries that avoid
                half-pixel offsets.

        Returns:
            list[Polygon]: List of Polygon objects representing the contours.
                Each Polygon has exterior coordinates tracing the contour path.

        Raises:
            ValueError: If input array is not 2-dimensional, has invalid dimensions
                (< 2x2), or has an unsupported dtype.

        Example:
            >>> import numpy as np
            >>> from dlup.geometry import marching_squares
            >>> # Using uint8 with level close to 1 for pixel-aligned boundaries
            >>> mask = np.zeros((5, 5), dtype=np.uint8)
            >>> mask[1:4, 1:4] = 1
            >>> contours = marching_squares.find_contours(mask, level=0.9)
            >>> len(contours)
            1
            >>> type(contours[0])
            <class 'dlup.geometry.Polygon'>
        )pbdoc");
}
