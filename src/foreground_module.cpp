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
#include <aifocore/concepts/numeric.h>
#include <aifocore/tiling/grid.h>
#include <aifocore/tiling/python/grid_wrapper.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dlup/backends/abstract.h"
#include "dlup/foreground.h"
#include "dlup/geometry/collection.h"
#include "dlup/slide_image.h"

namespace py = pybind11;

namespace dlup {

PYBIND11_MODULE(_foreground, m) {
  // Define a wrapper class that provides access to the foreground indices
  py::class_<ForegroundResult<int>>(m, "ForegroundResult")
      .def_property_readonly("foreground_indices",
                             [](const ForegroundResult<int>& self) {
                               return self.foreground_indices;
                             })
      .def_property_readonly(
          "unfiltered_grid",
          [](const ForegroundResult<int>& self) {
            return tiling::python::GridWrapper(*self.unfiltered_grid);
          })
      .def("to_grid", &ForegroundResult<int>::ToGrid);

  // Bind Foreground class
  py::class_<Foreground<int>>(m, "Foreground")
      .def_static(
          "filter_grid",
          [](const tiling::python::GridWrapper& grid_wrapper,
             const SlideImage& slide_image,
             const aifocore::Size<int, 2>& tile_size, double mpp,
             std::optional<double> threshold) {
            // Extract Grid<int> from GridWrapper
            const auto* grid = std::get_if<Grid<int>>(&grid_wrapper.GetGrid());
            if (!grid) {
              throw std::invalid_argument(
                  "Expected Grid<int>, but got Grid<double>.");
            }
            return Foreground<int>::FilterGrid(*grid, slide_image, tile_size,
                                               mpp, threshold);
          },
          py::arg("grid"), py::arg("slide_image"), py::arg("tile_size"),
          py::arg("mpp"), py::arg("threshold") = std::nullopt,
          "Filter a grid of tiles based on the SlideImage coverage threshold.")
      .def_static(
          "filter_grid",
          [](const tiling::python::GridWrapper& grid_wrapper,
             const geometry::GeometryCollection& collection,
             const aifocore::Size<int, 2>& tile_size, double scaling,
             std::optional<double> threshold) {
            // Extract Grid<int> from GridWrapper
            const auto* grid = std::get_if<Grid<int>>(&grid_wrapper.GetGrid());
            if (!grid) {
              throw std::invalid_argument(
                  "Expected Grid<int>, but got Grid<double>.");
            }
            return Foreground<int>::FilterGrid(*grid, collection, tile_size,
                                               scaling, threshold);
          },
          py::arg("grid"), py::arg("collection"), py::arg("tile_size"),
          py::arg("scaling"), py::arg("threshold") = std::nullopt,
          "Filter a grid of tiles based on the GeometryCollection coverage "
          "threshold.")
      .def_static(
          "filter_grid",
          [](const tiling::python::GridWrapper& grid_wrapper,
             const backends::AbstractSlideBackend& backend,
             const aifocore::Size<int, 2>& tile_size, double mpp,
             std::optional<double> threshold) {
            // Extract Grid<int> from GridWrapper
            const auto* grid = std::get_if<Grid<int>>(&grid_wrapper.GetGrid());
            if (!grid) {
              throw std::invalid_argument(
                  "Expected Grid<int>, but got Grid<double>.");
            }
            return Foreground<int>::FilterGrid(*grid, backend, tile_size, mpp,
                                               threshold);
          },
          py::arg("grid"), py::arg("backend"), py::arg("tile_size"),
          py::arg("mpp"), py::arg("threshold") = std::nullopt,
          "Filter a grid of tiles based on the AbstractSlideBackend coverage "
          "threshold.");
}

}  // namespace dlup
