#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geometry/base.h"
#include "geometry/box.h"
#include "geometry/collection.h"
#include "geometry/exceptions.h"
#include "geometry/factory.h"
#include "geometry/lazy_array.h"
#include "geometry/point.h"
#include "geometry/polygon.h"
#include "geometry/region.h"
namespace py = pybind11;

template class FactoryManager<Polygon>;
template class FactoryManager<Box>;
template class FactoryManager<Point>;

PYBIND11_MODULE(_geometry, m) {
  declare_base_geometry(m);
  declare_polygon(m);
  declare_box(m);
  declare_point(m);

  m.def("set_polygon_factory", &FactoryManager<Polygon>::setFactory, "Set the factory function for Polygons");
  m.def("set_box_factory", &FactoryManager<Box>::setFactory, "Set the factory function for Boxes");
  m.def("set_point_factory", &FactoryManager<Point>::setFactory, "Set the factory function for Points");

  declare_pybind_collection(m);
  declare_lazy_array<int>(m, "LazyArrayInt");
  declare_pybind_polygon_collection(m);
  declare_pybind_region(m);

  py::register_exception<GeometryError>(m, "GeometryError");
  py::register_exception<GeometryIntersectionError>(m, "GeometryIntersectionError");
  py::register_exception<GeometryTransformationError>(m, "GeometryTransformationError");
  py::register_exception<GeometryFactoryFunctionError>(m, "GeometryFactoryFunctionError");
  py::register_exception<GeometryNotFoundError>(m, "GeometryNotFoundError");
  py::register_exception<GeometryCoordinatesError>(m, "GeometryCoordinatesError");
}
