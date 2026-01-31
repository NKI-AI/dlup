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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "aifocore/status/result.h"
#include "vips_docstrings.h"

#include "dlup/backends/python/vips_wrapper.h"
#include "dlup/backends/vips.h"

namespace py = pybind11;

namespace dlup::backends::python::vips {

using Dimensions = Size<int, 2>;

PYBIND11_MODULE(_vips_backend, m) {
  InitializeVips();

  py::class_<dlup::backends::AbstractSlideBackend,
             std::shared_ptr<dlup::backends::AbstractSlideBackend>>(
      m, "AbstractSlideBackend")
      .def("get_best_level_for_downsample",
           &dlup::backends::AbstractSlideBackend::GetBestLevelForDownsample,
           get_best_level_for_downsample_doc)
      .def("set_spacing",
           [](dlup::backends::AbstractSlideBackend& self,
              const dlup::backends::Spacing& spacing) {
             aifocore::Status status = self.SetSpacing(spacing);
             if (!status.ok()) {
               throw std::runtime_error(std::string(status.message()));
             }
           })
      .def("close", &dlup::backends::AbstractSlideBackend::Close)
      .def_property_readonly(
          "level_count", &dlup::backends::AbstractSlideBackend::GetLevelCount,
          level_count_doc)
      .def_property_readonly("spacing",
                             &dlup::backends::AbstractSlideBackend::GetSpacing)
      .def_property_readonly(
          "dimensions", &dlup::backends::AbstractSlideBackend::GetDimensions,
          dimensions_doc)
      .def_property_readonly(
          "level_downsamples",
          [](const dlup::backends::AbstractSlideBackend& self) {
            return py::tuple(py::cast(self.GetLevelDownsamples()));
          })
      .def_property_readonly(
          "level_spacings",
          [](const dlup::backends::AbstractSlideBackend& self) {
            return py::tuple(py::cast(self.GetLevelSpacings()));
          })
      .def_property_readonly(
          "level_dimensions",
          [](const dlup::backends::AbstractSlideBackend& self) {
            return py::tuple(py::cast(self.GetLevelDimensions()));
          },
          level_dimensions_doc)
      .def_property_readonly(
          "slide_bounds", &dlup::backends::AbstractSlideBackend::GetSlideBounds)
      .def_property_readonly(
          "magnification",
          &dlup::backends::AbstractSlideBackend::GetMagnification)
      .def_property_readonly("vendor",
                             &dlup::backends::AbstractSlideBackend::GetVendor);

  py::class_<dlup::backends::VipsSlide, dlup::backends::AbstractSlideBackend,
             std::shared_ptr<dlup::backends::VipsSlide>>(m, "VipsSlide")

      .def(py::init<const std::string&, bool, bool, bool>(),
           py::arg("filename"), py::arg("load_with_openslide") = false,
           py::arg("rgb") = true, py::arg("apply_color_profile") = false)
      .def_property_readonly("spacing", &dlup::backends::VipsSlide::GetSpacing,
                             "Returns the spacing at the specified level.")
      .def("read_region", &dlup::backends::VipsSlide::ReadRegion,
           py::arg("coordinates"), py::arg("level"), py::arg("size"),
           read_region_doc)
      .def("get_thumbnail", &dlup::backends::VipsSlide::GetThumbnail,
           py::arg("size"), "Returns a thumbnail of the slide.")
      .def_property_readonly(
          "_libvips_version", [](const VipsSlide&) { return LibVipsVersion(); },
          "Returns the version of the VIPS library.");
}

}  // namespace dlup::backends::python::vips
