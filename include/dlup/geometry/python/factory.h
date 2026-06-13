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

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;

class FactoryGuard {
 public:
  FactoryGuard(nb::callable& factory_ref, nb::callable new_factory)
      : factory_ref_(factory_ref), original_factory_(factory_ref) {
    factory_ref_ = std::move(new_factory);
  }

  ~FactoryGuard() { factory_ref_ = original_factory_; }

 private:
  nb::callable& factory_ref_;
  nb::callable original_factory_;
};

template <typename T>
class FactoryManager {
 public:
  static void SetFactory(nb::callable factory) {
    factoryFunction() = std::move(factory);
  }

  static nb::object CallFactoryFunction(const std::shared_ptr<T>& object) {
    return InvokeFactoryFunction(factoryFunction(), object);
  }

  static FactoryGuard CreateFactoryGuard(nb::callable factory) {
    return FactoryGuard(factoryFunction(), std::move(factory));
  }

  template <typename U>
  static void SetAndCreateFactoryGuard(nb::callable factory) {
    SetFactory(factory);
    CreateFactoryGuard(std::move(factory));
  }

 private:
  static nb::callable& factoryFunction() {
    static nb::callable instance;
    return instance;
  }

  static nb::object InvokeFactoryFunction(nb::callable factoryFunction,
                                          const std::shared_ptr<T>& object) {
    if (!factoryFunction.is_valid() ||
        !PyCallable_Check(factoryFunction.ptr())) {  // NOLINT(*)
      return nb::cast(object);
    }

    try {
      nb::object result = factoryFunction(object);
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
