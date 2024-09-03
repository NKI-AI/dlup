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
  py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
      .def("set_field", &BaseGeometry::setField)
      .def("get_field", &BaseGeometry::getField)
      .def_property_readonly("fields", &BaseGeometry::getFields)
      .def_property_readonly("pointer_id", &BaseGeometry::getPointerId);

  py::class_<Polygon, BaseGeometry, std::shared_ptr<Polygon>>(m, "Polygon")
      .def(py::init<>())
      .def(py::init<const BoostPolygon &>())
      .def(py::init<const std::vector<std::pair<double, double>> &,
                    const std::vector<std::vector<std::pair<double, double>>> &>())
      .def(py::init([](const std::shared_ptr<Polygon> &p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const Polygon &other) {
        // Explicitly copy parameters when copying the polygon
        auto newPolygon = std::make_shared<Polygon>(*other.polygon_);
        newPolygon->parameters_ = other.parameters_; // Copy the parameters
        return newPolygon;
      }))
      .def("set_exterior", &Polygon::setExterior)
      .def("set_interiors", &Polygon::setInteriors)
      .def("get_exterior", &Polygon::getExterior)
      .def("get_exterior_iterator",
           [](Polygon &self) {
             return py::make_iterator(self.getExteriorAsIterator().begin(), self.getExteriorAsIterator().end());
           })
      .def("get_interiors_iterator",
           [](Polygon &self) {
             return py::make_iterator(self.getInteriorAsIterator().begin(), self.getInteriorAsIterator().end());
           })
      .def("scale", &Polygon::scale, py::arg("scaling"))
      .def("get_interiors", &Polygon::getInteriors)
      .def("correct_orientation", &Polygon::correctIfNeeded)
      .def("simplify", &Polygon::simplifyPolygon)
      .def("contains", &Polygon::contains, py::arg("other"),
           "Check if the polygon fully contains another polygon. Does not check if the fields are equal")
      .def("make_valid", &Polygon::makeValid,
           "Make the polygon valid by removing self-intersections and duplicate points")
      .def("equals", &Polygon::equals, py::arg("other"),
           "Check if the polygon is equal to another polygon. Checks if the fields are equal.")
      .def_property_readonly("wkt", &Polygon::toWkt)
      .def_property_readonly("is_valid", &Polygon::isValid)
      .def_property_readonly("area", &Polygon::getArea);

  py::class_<Box, BaseGeometry, std::shared_ptr<Box>>(m, "Box")
      .def(py::init<>())
      .def(py::init<const BoostBox &>())
      .def(py::init<const std::array<double, 2> &, const std::array<double, 2> &>())
      .def(py::init([](const std::shared_ptr<Box> &p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const Box &other) {
        // Explicitly copy parameters when copying the Box
        auto newBox = std::make_shared<Box>(*other.box_);
        newBox->parameters_ = other.parameters_; // Copy the parameters
        return newBox;
      }))
      .def("as_polygon", &Box::asPolygonPyObject, "Convert the box to a polygon")
      .def("scale", &Box::scale, py::arg("scaling"), "Scale the box in-place by a factor")

      .def_property_readonly("coordinates", &Box::getCoordinates,
                             "Get the top-left coordinates of the box as an (x, y) tuple")
      .def_property_readonly("size", &Box::getSize, "Get the size of the box as an (h, w) tuple")
      .def_property_readonly("area", &Box::getArea)
      .def_property_readonly("wkt", &Box::toWkt, "Get the WKT representation of the box");

  py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
      .def(py::init<>())
      .def(py::init<const BoostPoint &>())
      .def(py::init<double, double>())
      .def(py::init([](const std::shared_ptr<Point> &p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const Point &other) {
        // Explicitly copy parameters when copying the polygon
        auto newPoint = std::make_shared<Point>(*other.point_);
        newPoint->parameters_ = other.parameters_; // Copy the parameters
        return newPoint;
      }))
      .def_property_readonly("coordinates", &Point::getCoordinates,
                             "Get the coordinates of the point as an (x, y) tuple")
      .def_property_readonly("x", &Point::getX, "Get the X coordinate")
      .def_property_readonly("y", &Point::getY, "Get the Y coordinate")
      .def("distance_to", &Point::distanceTo, py::arg("other"), "Calculate the distance to another point")
      .def("equals", &Point::equals, py::arg("other"), "Check if the point is equal to another point")
      .def("within", &Point::within, py::arg("polygon"), "Check if the point is within a polygon")
      .def("scale", &Point::scale, py::arg("scaling"), "Scale the point in-place point by a factor")
      .def_property_readonly("wkt", &Point::toWkt, "Get the WKT representation of the point");

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
