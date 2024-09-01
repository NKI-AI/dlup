#ifndef DLUP_GEOMETRY_LAZY_ARRAY_H
#define DLUP_GEOMETRY_LAZY_ARRAY_H

#include <functional>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename T>
class LazyArray {
  public:
  using ComputeFunction = std::function<py::array_t<T>()>;

  LazyArray(ComputeFunction compute_func) : compute_func_(std::move(compute_func)), computed_(false) {}

  py::array_t<T> numpy() {
    if (!computed_) {
      data_ = compute_func_();
      computed_ = true;
    }
    return data_;
  }

  py::array_t<T> operator*() { return numpy(); }

  // Changed this method to return py::array_t<T> directly
  py::array_t<T> py_numpy() { return numpy(); }

  private:
  ComputeFunction compute_func_;
  py::array_t<T> data_;
  bool computed_;
};

template <typename T>
void declare_lazy_array(py::module &m, const std::string &type_name) {
  py::class_<LazyArray<T>>(m, type_name.c_str())
      .def(py::init<typename LazyArray<T>::ComputeFunction>())
      .def("numpy", &LazyArray<T>::py_numpy)
      .def("__array__", &LazyArray<T>::py_numpy)
      .def("__repr__", [](const LazyArray<T> &) { return "<LazyArray: use numpy() or __array__() to compute>"; });
}

#endif // DLUP_GEOMETRY_LAZY_ARRAY_H