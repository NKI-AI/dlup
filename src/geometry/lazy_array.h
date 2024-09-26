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

  LazyArray(ComputeFunction compute_func, std::vector<py::ssize_t> shape)
      : compute_func_(std::move(compute_func)), computed_(false), shape_(std::move(shape)) {}

  py::array_t<T> numpy() const {
    if (!computed_) {
      data_ = compute_func_();
      computed_ = true;
    }
    return data_;
  }

  py::array_t<T> operator*() const { return numpy(); }

  py::array_t<T> py_numpy() const { return numpy(); }

  std::vector<py::ssize_t> shape() const { return shape_; }

  LazyArray<T> transpose(const std::vector<py::ssize_t> &axes = {}) const {
    std::vector<py::ssize_t> new_shape(shape_);
    if (axes.empty()) {
      std::reverse(new_shape.begin(), new_shape.end());
    } else {
      for (size_t i = 0; i < axes.size(); ++i) {
        new_shape[i] = shape_[axes[i]];
      }
    }

    return LazyArray<T>(
        [this, axes]() { return this->numpy().attr("transpose")(axes).template cast<py::array_t<T>>(); }, new_shape);
  }

  LazyArray<T> reshape(const std::vector<py::ssize_t> &new_shape) const {
    return LazyArray<T>([this, new_shape]() { return this->numpy().reshape(new_shape); }, new_shape);
  }

  LazyArray<T> operator+(const LazyArray<T> &other) const {
    return LazyArray<T>([this, &other]() { return this->numpy() + other.numpy(); }, shape_);
  }

  LazyArray<T> operator-(const LazyArray<T> &other) const {
    return LazyArray<T>([this, &other]() { return this->numpy() - other.numpy(); }, shape_);
  }

  LazyArray<T> multiply(const LazyArray<T> &other) const {
    return LazyArray<T>([this, &other]() { return this->numpy() * other.numpy(); }, shape_);
  }

  LazyArray<T> operator/(const LazyArray<T> &other) const {
    return LazyArray<T>([this, &other]() { return this->numpy() / other.numpy(); }, shape_);
  }

  private:
  ComputeFunction compute_func_;
  mutable py::array_t<T> data_;
  mutable bool computed_;
  std::vector<py::ssize_t> shape_;
};

template <typename T>
void declare_lazy_array(py::module &m, const std::string &type_name) {
  py::class_<LazyArray<T>>(m, type_name.c_str())
      .def(py::init<typename LazyArray<T>::ComputeFunction, std::vector<py::ssize_t>>())
      .def("numpy", &LazyArray<T>::py_numpy)
      .def("shape", &LazyArray<T>::shape)
      .def("transpose", &LazyArray<T>::transpose, py::arg("axes") = std::vector<py::ssize_t>())
      .def("reshape", &LazyArray<T>::reshape)
      .def("__array__", &LazyArray<T>::py_numpy)
      .def("__repr__", [](const LazyArray<T> &) { return "<LazyArray: use numpy() or __array__() to compute>"; })
      .def("__add__", &LazyArray<T>::operator+)
      .def("__sub__", &LazyArray<T>::operator-)
      .def("__mul__", &LazyArray<T>::multiply)
      .def("__truediv__", &LazyArray<T>::operator/);
}

#endif // DLUP_GEOMETRY_LAZY_ARRAY_H
