#ifndef DLUP_GEOMETRY_FACTORY_H
#define DLUP_GEOMETRY_FACTORY_H
#pragma once

#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// FactoryGuard class definition
class FactoryGuard {
  public:
  FactoryGuard(py::function &factory_ref, py::function new_factory)
      : factory_ref_(factory_ref), original_factory_(factory_ref) {
    factory_ref_ = new_factory;
  }

  ~FactoryGuard() { factory_ref_ = original_factory_; }

  private:
  py::function &factory_ref_;
  py::function original_factory_;
};

// Template class to manage factory functions
template <typename T>
class FactoryManager {
  public:
  static void setFactory(py::function factory) { factoryFunction() = std::move(factory); }

  static py::object callFactoryFunction(const std::shared_ptr<T> &object) {
    return invokeFactoryFunction(factoryFunction(), object);
  }

  static FactoryGuard createFactoryGuard(py::function factory) { return FactoryGuard(factoryFunction(), factory); }

  // New method to streamline setting factories and creating guards
  template <typename U>
  static void setAndCreateFactoryGuard(py::function factory) {
    setFactory(factory);
    createFactoryGuard(factory);
  }

  private:
  static py::function &factoryFunction() {
    static py::function instance;
    return instance;
  }

  static py::object invokeFactoryFunction(py::function factoryFunction, const std::shared_ptr<T> &object) {
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
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Exception in factory function: ") + e.what());
    } catch (...) {
      throw std::runtime_error("Unknown exception in factory function");
    }
  }
};

#endif // DLUP_GEOMETRY_FACTORY_H
