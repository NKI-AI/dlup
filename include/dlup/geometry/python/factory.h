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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_PYTHON_FACTORY_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_PYTHON_FACTORY_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <utility>

namespace py = pybind11;

// FactoryGuard class definition
class FactoryGuard {
 public:
  FactoryGuard(py::function& factory_ref, py::function new_factory)
      : factory_ref_(factory_ref), original_factory_(factory_ref) {
    factory_ref_ = new_factory;
  }

  ~FactoryGuard() { factory_ref_ = original_factory_; }

 private:
  py::function& factory_ref_;
  py::function original_factory_;
};

// Template class to manage factory functions
template <typename T>
class FactoryManager {
 public:
  static void SetFactory(py::function factory) {
    factoryFunction() = std::move(factory);
  }

  static py::object CallFactoryFunction(const std::shared_ptr<T>& object) {
    return InvokeFactoryFunction(factoryFunction(), object);
  }

  static FactoryGuard CreateFactoryGuard(py::function factory) {
    return FactoryGuard(factoryFunction(), factory);
  }

  // New method to streamline setting factories and creating guards
  template <typename U>
  static void SetAndCreateFactoryGuard(py::function factory) {
    SetFactory(factory);
    CreateFactoryGuard(factory);
  }

 private:
  static py::function& factoryFunction() {
    static py::function instance;
    return instance;
  }

  static py::object InvokeFactoryFunction(py::function factoryFunction,
                                          const std::shared_ptr<T>& object) {
    if (!factoryFunction || !PyCallable_Check(factoryFunction.ptr())) {
      return py::cast(object);
    }

    try {
      py::object result = factoryFunction(object);
      if (!result.is_none()) {
        return result;
      } else {
        throw std::runtime_error("Factory function returned null object");
      }
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Exception in factory function: ") +
                               e.what());
    } catch (...) {
      throw std::runtime_error("Unknown exception in factory function");
    }
  }
};

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_PYTHON_FACTORY_H_
