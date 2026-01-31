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
#ifndef AIFO_DLUP_INCLUDE_DLUP_BACKENDS_PYTHON_VIPS_WRAPPER_H_
#define AIFO_DLUP_INCLUDE_DLUP_BACKENDS_PYTHON_VIPS_WRAPPER_H_

#include <vips/vips.h>
#include <vips/vips8>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dlup::backends::python::vips {

// TODO(jonasteuwen): This file needs to go in favor of aifo/utilities/vips.h
void InitializeVips() {
  if (VIPS_INIT("vips_python_bindings")) {
    vips_error_exit(nullptr);
  }
  // Force single-threaded mode (1 = single thread, 0 = auto-detect)
  vips_concurrency_set(1);
  vips_cache_set_max(0);

// Optional: switch on leak detection
#ifdef VIPS_LEAK_CHECK
  vips_leak_set(TRUE);
#endif
}

py::tuple LibVipsVersion() {
  return py::make_tuple(vips_version(0), vips_version(1), vips_version(2));
}

}  // namespace dlup::backends::python::vips

#endif  // AIFO_DLUP_INCLUDE_DLUP_BACKENDS_PYTHON_VIPS_WRAPPER_H_
