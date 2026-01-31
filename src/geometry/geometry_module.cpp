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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "dlup/geometry/base.h"
#include "dlup/geometry/box.h"
#include "dlup/geometry/collection.h"
#include "dlup/geometry/exceptions.h"
#include "dlup/geometry/lazy_array.h"
#include "dlup/geometry/marching_squares.h"
#include "dlup/geometry/point.h"
#include "dlup/geometry/polygon.h"
#include "dlup/geometry/python/factory.h"
#include "dlup/geometry/region.h"

#include "aifocore/math/ndarray.h"

namespace py = pybind11;

template class FactoryManager<dlup::geometry::Polygon>;
template class FactoryManager<dlup::geometry::Box>;
template class FactoryManager<dlup::geometry::Point>;

inline void DeclarePoint(py::module& m) {
  py::class_<dlup::geometry::Point, dlup::geometry::BaseGeometry,
             std::shared_ptr<dlup::geometry::Point>>(m, "Point")
      .def(py::init<>())
      .def(py::init<const dlup::geometry::BoostPoint&>())
      .def(py::init<double, double>())
      .def(py::init([](const std::shared_ptr<dlup::geometry::Point>& p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const dlup::geometry::Point& other) {
        // Explicitly copy parameters when copying the polygon
        auto newPoint = std::make_shared<dlup::geometry::Point>(*other.point_);
        newPoint->parameters_ = other.parameters_;  // Copy the parameters
        return newPoint;
      }))
      .def_property_readonly(
          "coordinates", &dlup::geometry::Point::GetCoordinates,
          "Get the coordinates of the point as an (x, y) tuple")
      .def_property_readonly("x", &dlup::geometry::Point::GetX,
                             "Get the X coordinate")
      .def_property_readonly("y", &dlup::geometry::Point::GetY,
                             "Get the Y coordinate")
      .def("distance_to", &dlup::geometry::Point::DistanceTo, py::arg("other"),
           "Calculate the distance to another point")
      .def("equals", &dlup::geometry::Point::Equals, py::arg("other"),
           "Check if the point is equal to another point")
      .def("within", &dlup::geometry::Point::Within, py::arg("polygon"),
           "Check if the point is within a polygon")
      .def("scale", &dlup::geometry::Point::Scale, py::arg("scaling"),
           "Scale the point in-place point by a factor")
      .def_property_readonly("wkt", &dlup::geometry::Point::ToWkt,
                             "Get the WKT representation of the point");
}

inline void DeclarePolygon(py::module& m) {
  py::class_<dlup::geometry::Polygon, dlup::geometry::BaseGeometry,
             std::shared_ptr<dlup::geometry::Polygon>>(m, "Polygon")
      .def(py::init<>())
      .def(py::init<const dlup::geometry::BoostPolygon&>())
      .def(py::init<
           const std::vector<std::pair<double, double>>&,
           const std::vector<std::vector<std::pair<double, double>>>&>())
      .def(py::init([](const std::shared_ptr<dlup::geometry::Polygon>& p) {
        // Share the same C++ object, not creating a new one
        return p;
      }))
      .def(py::init([](const dlup::geometry::Polygon& other) {
        // Explicitly copy parameters when copying the polygon
        auto new_polygon =
            std::make_shared<dlup::geometry::Polygon>(*other.polygon_);
        new_polygon->parameters_ = other.parameters_;  // Copy the parameters
        return new_polygon;
      }))
      .def("set_exterior", &dlup::geometry::Polygon::SetExterior)
      .def("set_interiors", &dlup::geometry::Polygon::SetInteriors)
      .def("get_exterior", &dlup::geometry::Polygon::GetExterior)
      .def("get_exterior_iterator",
           [](dlup::geometry::Polygon& self) {
             return py::make_iterator(self.GetExteriorAsIterator().begin(),
                                      self.GetExteriorAsIterator().end());
           })
      .def("get_interiors_iterator",
           [](dlup::geometry::Polygon& self) {
             return py::make_iterator(self.GetInteriorAsIterator().begin(),
                                      self.GetInteriorAsIterator().end());
           })
      .def("scale", &dlup::geometry::Polygon::Scale, py::arg("scaling"))
      .def("get_interiors", &dlup::geometry::Polygon::GetInteriors)
      .def("correct_orientation", &dlup::geometry::Polygon::CorrectIfNeeded)
      .def("simplify", &dlup::geometry::Polygon::SimplifyPolygon)
      .def("contains", &dlup::geometry::Polygon::Contains, py::arg("other"),
           "Check if the polygon fully contains another polygon. Does not "
           "check if the fields are equal")
      .def("make_valid", &dlup::geometry::Polygon::MakeValid,
           "Make the polygon valid by removing self-intersections and "
           "duplicate points")
      .def("equals", &dlup::geometry::Polygon::Equals, py::arg("other"),
           "Check if the polygon is equal to another polygon. Checks if the "
           "fields are equal.")
      .def_property_readonly("wkt", &dlup::geometry::Polygon::ToWkt)
      .def_property_readonly("is_valid", &dlup::geometry::Polygon::IsValid)
      .def_property_readonly("area", &dlup::geometry::Polygon::GetArea)
      .def_property_readonly("bounding_box",
                             &dlup::geometry::Polygon::GetBoundingBox);
}

void DeclareCollection(py::module& m) {
  py::class_<dlup::geometry::GeometryCollection,
             std::shared_ptr<dlup::geometry::GeometryCollection>>(
      m, "GeometryCollection")
      .def(py::init<>())
      .def(py::pickle(
          [](const dlup::geometry::GeometryCollection& collection) {
            // Serialize into a dictionary-like Python object
            py::dict state;
            state["polygons"] = collection.GetPolygons();
            state["points"] = collection.GetPoints();
            state["boxes"] = collection.GetBoxes();
            state["rois"] = collection.GetRois();
            state["rtree_invalidated"] = collection.IsRTreeInvalidated();
            return state;
          },
          [](py::dict state) {
            bool was_rtree_invalidated =
                state["rtree_invalidated"].cast<bool>();
            auto collection =
                std::make_shared<dlup::geometry::GeometryCollection>();

            for (const auto& polygon :
                 state["polygons"]
                     .cast<std::vector<
                         std::shared_ptr<dlup::geometry::Polygon>>>())
              collection->AddPolygon(polygon);
            for (const auto& point :
                 state["points"]
                     .cast<
                         std::vector<std::shared_ptr<dlup::geometry::Point>>>())
              collection->AddPoint(point);
            for (const auto& box :
                 state["boxes"]
                     .cast<std::vector<std::shared_ptr<dlup::geometry::Box>>>())
              collection->AddBox(box);
            for (const auto& roi :
                 state["rois"]
                     .cast<std::vector<
                         std::shared_ptr<dlup::geometry::Polygon>>>())
              collection->AddRoi(roi);

            if (!was_rtree_invalidated) {
              collection->RebuildRTree();
            }

            return collection;
          }))

      .def("add_polygon", &dlup::geometry::GeometryCollection::AddPolygon)
      .def("add_roi", &dlup::geometry::GeometryCollection::AddRoi)
      .def("add_point", &dlup::geometry::GeometryCollection::AddPoint)
      .def("add_box", &dlup::geometry::GeometryCollection::AddBox)
      .def_property_readonly("num_polygons",
                             &dlup::geometry::GeometryCollection::NumPolygons)
      .def_property_readonly("num_rois",
                             &dlup::geometry::GeometryCollection::NumRois)
      .def_property_readonly("num_points",
                             &dlup::geometry::GeometryCollection::NumPoints)
      .def_property_readonly("num_boxes",
                             &dlup::geometry::GeometryCollection::NumBoxes)
      .def_property_readonly("has_rois",
                             &dlup::geometry::GeometryCollection::HasRois)

      // Overload remove_polygon to handle both object and index
      .def("remove_polygon",
           py::overload_cast<const std::shared_ptr<dlup::geometry::Polygon>&>(
               &dlup::geometry::GeometryCollection::RemovePolygon),
           "Remove a polygon by passing the Polygon object")
      .def("remove_polygon",
           py::overload_cast<size_t>(
               &dlup::geometry::GeometryCollection::RemovePolygon),
           "Remove a polygon by its index")
      .def("remove_box",
           py::overload_cast<const std::shared_ptr<dlup::geometry::Box>&>(
               &dlup::geometry::GeometryCollection::RemoveBox),
           "Remove a box by passing the Box object")
      .def("remove_box",
           py::overload_cast<size_t>(
               &dlup::geometry::GeometryCollection::RemoveBox),
           "Remove a box by its index")
      .def("remove_roi",
           py::overload_cast<const std::shared_ptr<dlup::geometry::Polygon>&>(
               &dlup::geometry::GeometryCollection::RemoveRoi),
           "Remove an ROI by passing the ROI object")
      .def("remove_roi",
           py::overload_cast<size_t>(
               &dlup::geometry::GeometryCollection::RemoveRoi),
           "Remove an ROI by its index")
      .def("reindex_polygons",
           &dlup::geometry::GeometryCollection::ReindexPolygons)
      .def(
          "sort_polygons",
          [](dlup::geometry::GeometryCollection& self,
             const py::function& key_func, bool reverse) {
            self.SortPolygons([&key_func, reverse](const auto& a,
                                                   const auto& b) {
              py::object key_a = key_func(a);
              py::object key_b = key_func(b);

              if (py::isinstance<py::str>(key_a) &&
                  py::isinstance<py::str>(key_b)) {
                return reverse ? (key_a.cast<std::string>() >
                                  key_b.cast<std::string>())
                               : (key_a.cast<std::string>() <
                                  key_b.cast<std::string>());
              } else if (py::isinstance<py::float_>(key_a) &&
                         py::isinstance<py::float_>(key_b)) {
                return reverse ? (key_a.cast<double>() > key_b.cast<double>())
                               : (key_a.cast<double>() < key_b.cast<double>());
              } else if (py::isinstance<py::int_>(key_a) &&
                         py::isinstance<py::int_>(key_b)) {
                return reverse ? (key_a.cast<int>() > key_b.cast<int>())
                               : (key_a.cast<int>() < key_b.cast<int>());
              } else if (py::isinstance<py::none>(key_a) &&
                         py::isinstance<py::none>(key_b)) {
                return false;
              } else {
                throw std::invalid_argument(
                    "Unsupported key type for sorting.");
              }
            });
          },
          "Sort polygons by a custom key function")
      .def("simplify_polygons",
           &dlup::geometry::GeometryCollection::SimplifyPolygons)
      .def("__len__", &dlup::geometry::GeometryCollection::Size)

      // Overload remove_point to handle both object and index
      .def("remove_point",
           py::overload_cast<const std::shared_ptr<dlup::geometry::Point>&>(
               &dlup::geometry::GeometryCollection::RemovePoint),
           "Remove a point by passing the Point object")
      .def("remove_point",
           py::overload_cast<size_t>(
               &dlup::geometry::GeometryCollection::RemovePoint),
           "Remove a point by its index")
      .def("read_region", &dlup::geometry::GeometryCollection::ReadRegion)
      .def("rebuild_rtree", &dlup::geometry::GeometryCollection::RebuildRTree,
           "Rebuild the R-tree index manually")
      .def("scale", &dlup::geometry::GeometryCollection::Scale,
           "Scale all geometries by a factor")
      .def("set_offset", &dlup::geometry::GeometryCollection::SetOffset,
           "Set an offset for all geometries")
      .def_property_readonly(
          "rtree_invalidated",
          &dlup::geometry::GeometryCollection::IsRTreeInvalidated)
      .def_property_readonly("pointer_id",
                             &dlup::geometry::GeometryCollection::GetPointerId)
      .def_property_readonly(
          "bounding_box",
          &dlup::geometry::GeometryCollection::ComputeBoundingBox)
      .def_property_readonly(
          "polygons",
          [](dlup::geometry::GeometryCollection& self) {
            py::list py_polygons;
            for (const auto& polygon : self.GetPolygons()) {
              py::object processed_polygon =
                  FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                      polygon);
              py_polygons.append(processed_polygon);
            }
            return py_polygons;
          })
      .def_property_readonly(
          "rois",
          [](dlup::geometry::GeometryCollection& self) {
            py::list py_rois;
            for (const auto& roi : self.GetRois()) {
              py::object processed_roi =
                  FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                      roi);
              py_rois.append(processed_roi);
            }
            return py_rois;
          })
      .def_property_readonly(
          "points",
          [](dlup::geometry::GeometryCollection& self) {
            py::list py_points;
            for (const auto& point : self.GetPoints()) {
              py::object processed_point =
                  FactoryManager<dlup::geometry::Point>::CallFactoryFunction(
                      point);
              py_points.append(processed_point);
            }
            return py_points;
          })
      .def_property_readonly(
          "boxes",
          [](dlup::geometry::GeometryCollection& self) {
            py::list py_boxes;
            for (const auto& box : self.GetBoxes()) {
              py::object processed_box =
                  FactoryManager<dlup::geometry::Box>::CallFactoryFunction(box);
              py_boxes.append(processed_box);
            }
            return py_boxes;
          })
      .def_property_readonly("index_map",
                             &dlup::geometry::GeometryCollection::GetIndexMap);
}

void DeclarePolygonCollection(py::module& m) {
  py::class_<dlup::geometry::PolygonCollection,
             std::shared_ptr<dlup::geometry::PolygonCollection>>(
      m, "PolygonCollection")
      .def("get_geometries",
           [](dlup::geometry::PolygonCollection& self) {
             py::list py_polygons;
             for (const auto& polygon : self.GetGeometries()) {
               py::object processed_polygon =
                   FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                       polygon);
               py_polygons.append(processed_polygon);
             }
             return py_polygons;
           })
      .def("to_mask", &dlup::geometry::PolygonCollection::ToMask,
           py::arg("default_value") = 0);
}

inline void DeclareBaseGeometry(py::module& m) {
  py::class_<dlup::geometry::BaseGeometry,
             std::shared_ptr<dlup::geometry::BaseGeometry>>(m, "BaseGeometry")
      .def("set_field", &dlup::geometry::BaseGeometry::SetField)
      .def(
          "get_field",
          [](dlup::geometry::BaseGeometry& self,
             const std::string& name) -> py::object {
            auto field = self.GetField(name);
            if (!field) {
              return py::none();
            }
            return std::visit(
                [](const auto& value) -> py::object { return py::cast(value); },
                *field);
          })
      .def(
          "clone",
          [](const dlup::geometry::BaseGeometry& self) {
            auto cloned = self.Clone();
            // Use the appropriate FactoryManager
            // to create the Python equivalent
            if (auto point =
                    std::dynamic_pointer_cast<dlup::geometry::Point>(cloned)) {
              return FactoryManager<dlup::geometry::Point>::CallFactoryFunction(
                  point);
            } else if (auto polygon =
                           std::dynamic_pointer_cast<dlup::geometry::Polygon>(
                               cloned)) {
              return FactoryManager<
                  dlup::geometry::Polygon>::CallFactoryFunction(polygon);
            } else if (auto box =
                           std::dynamic_pointer_cast<dlup::geometry::Box>(
                               cloned)) {
              return FactoryManager<dlup::geometry::Box>::CallFactoryFunction(
                  box);
            } else {
              throw std::runtime_error("Unsupported geometry type in clone");
            }
          },
          "Create a deep copy of the geometry")
      .def_property_readonly("fields", &dlup::geometry::BaseGeometry::GetFields)
      .def_property_readonly("pointer_id",
                             &dlup::geometry::BaseGeometry::GetPointerId);
}

inline void DeclareBox(py::module& m) {
  py::class_<dlup::geometry::Box, dlup::geometry::BaseGeometry,
             std::shared_ptr<dlup::geometry::Box>>(m, "Box")
      .def(py::init<>())
      .def(py::init<const dlup::geometry::BoostBox&>())
      .def(py::init<const std::array<double, 2>&,
                    const std::array<double, 2>&>())
      .def(py::init(
          [](const std::shared_ptr<dlup::geometry::Box>& p) { return p; }))
      .def(py::init([](const dlup::geometry::Box& other) {
        auto newBox = std::make_shared<dlup::geometry::Box>(*other.box_);
        newBox->parameters_ = other.parameters_;
        return newBox;
      }))
      .def(
          "as_polygon",
          [](const dlup::geometry::Box& box) {
            auto polygon = box.AsPolygon();  // Call the AsPolygon method
            return FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                polygon);  // Apply the FactoryManager logic
          },
          "Convert the box to a polygon")
      .def("scale", &dlup::geometry::Box::Scale, py::arg("scaling"),
           "Scale the box in-place by a factor")
      .def_property_readonly(
          "coordinates",
          [](const dlup::geometry::Box& self) {
            auto coords = self.GetCoordinates();
            return py::make_tuple(coords[0], coords[1]);
          },
          "Get the top-left coordinates of the box as an (x, y) tuple")
      .def_property_readonly(
          "size",
          [](const dlup::geometry::Box& self) {
            auto size = self.GetSize();
            return py::make_tuple(size[0], size[1]);
          },
          "Get the size of the box as an (h, w) tuple")
      .def_property_readonly("area", &dlup::geometry::Box::GetArea)
      .def_property_readonly("wkt", &dlup::geometry::Box::ToWkt,
                             "Get the WKT representation of the box");
}

template <typename T>
void DeclareLazyArray(py::module& m, const std::string& type_name) {
  py::class_<LazyArray<T>>(m, type_name.c_str())
      .def(py::init<typename LazyArray<T>::ComputeFunction,
                    std::vector<std::size_t>>())
      .def("numpy",
           [](const LazyArray<T>& arr) -> py::array_t<T> {
             const auto& data = arr.data();
             const auto& shape = arr.shape();

             // Create a NumPy array that owns its data
             auto result = py::array_t<T>(
                 std::vector<py::ssize_t>(shape.begin(), shape.end()));

             // Copy the data from vector to the NumPy array
             auto result_data = result.mutable_data();
             std::copy(data.begin(), data.end(), result_data);

             return result;
           })
      .def("shape", &LazyArray<T>::shape)
      .def("__repr__", [](const LazyArray<T>&) {
        return "<LazyArray: use numpy() to compute>";
      });
}

void DeclareRegion(py::module& m) {
  py::class_<dlup::geometry::AnnotationRegion,
             std::shared_ptr<dlup::geometry::AnnotationRegion>>(
      m, "AnnotationRegion")
      .def(py::init<std::function<dlup::geometry::AnnotationRegion()>, bool>())
      .def(py::init<std::vector<std::shared_ptr<dlup::geometry::Polygon>>,
                    std::vector<std::shared_ptr<dlup::geometry::Polygon>>,
                    std::vector<std::shared_ptr<dlup::geometry::Box>>,
                    std::vector<std::shared_ptr<dlup::geometry::Point>>,
                    std::tuple<int, int>, bool>())
      .def_property_readonly("polygons",
                             &dlup::geometry::AnnotationRegion::GetPolygons)
      .def_property_readonly("rois", &dlup::geometry::AnnotationRegion::GetRois)
      .def_property_readonly(
          "boxes",
          [](dlup::geometry::AnnotationRegion& self) {
            auto boxes = self.GetBoxes();
            py::list py_boxes;
            for (const auto& box : boxes) {
              py_boxes.append(
                  FactoryManager<dlup::geometry::Box>::CallFactoryFunction(
                      box));
            }
            return py_boxes;
          })

      .def_property_readonly(
          "points",
          [](dlup::geometry::AnnotationRegion& self) {
            auto points = self.GetPoints();
            py::list py_points;
            for (const auto& point : points) {
              py_points.append(
                  FactoryManager<dlup::geometry::Point>::CallFactoryFunction(
                      point));
            }
            return py_points;
          })
      .def_property_readonly("has_rois",
                             &dlup::geometry::AnnotationRegion::HasRois);
}

namespace {

template <typename T, int Flags>
std::vector<std::shared_ptr<dlup::geometry::Polygon>> ProcessArrayForContours(
    const py::array_t<T, Flags>& image_in, double level) {
  auto image =
      py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
          image_in);
  if (!image) {
    throw std::invalid_argument(
        "Input array could not be converted to the required format (C-style "
        "contiguous).");
  }
  py::buffer_info buf_info = image.request();

  if (buf_info.ndim != 2) {
    throw std::invalid_argument("Input array must be 2-dimensional");
  }

  const std::size_t height = static_cast<std::size_t>(buf_info.shape[0]);
  const std::size_t width = static_cast<std::size_t>(buf_info.shape[1]);

  // Array is guaranteed to be C-contiguous due to py::array::c_style
  if constexpr (std::is_same_v<T, double>) {
    aifocore::math::NDArrayView<double, 2> ndarray_view(
        static_cast<double*>(buf_info.ptr), {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  } else {
    std::vector<double> converted_data(height * width);
    const T* src = static_cast<T*>(buf_info.ptr);

    // Safe to iterate linearly since array is C-contiguous
    for (std::size_t i = 0; i < height * width; ++i) {
      converted_data[i] = static_cast<double>(src[i]);
    }

    aifocore::math::NDArrayView<double, 2> ndarray_view(converted_data.data(),
                                                        {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  }
}

py::list FindContoursPython(const py::array& image, double level) {
  std::vector<std::shared_ptr<dlup::geometry::Polygon>> contours;

  if (py::isinstance<py::array_t<uint8_t>>(image)) {
    contours =
        ProcessArrayForContours(image.cast<py::array_t<uint8_t>>(), level);
  } else if (py::isinstance<py::array_t<int32_t>>(image)) {
    contours =
        ProcessArrayForContours(image.cast<py::array_t<int32_t>>(), level);
  } else if (py::isinstance<py::array_t<float>>(image)) {
    contours = ProcessArrayForContours(image.cast<py::array_t<float>>(), level);
  } else if (py::isinstance<py::array_t<double>>(image)) {
    contours =
        ProcessArrayForContours(image.cast<py::array_t<double>>(), level);
  } else {
    throw std::invalid_argument(
        "Input array must be of type uint8, int32, float32, or float64");
  }

  // Apply factory pattern to convert to Python Polygon objects
  py::list result;
  for (const auto& polygon : contours) {
    py::object py_polygon =
        FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(polygon);
    result.append(py_polygon);
  }

  return result;
}

}  // namespace

PYBIND11_MODULE(_geometry, m) {
  DeclareBaseGeometry(m);
  DeclarePolygon(m);
  DeclareBox(m);
  DeclarePoint(m);

  m.def("set_polygon_factory",
        &FactoryManager<dlup::geometry::Polygon>::SetFactory,
        "Set the factory function for Polygons");
  m.def("set_box_factory", &FactoryManager<dlup::geometry::Box>::SetFactory,
        "Set the factory function for Boxes");
  m.def("set_point_factory", &FactoryManager<dlup::geometry::Point>::SetFactory,
        "Set the factory function for Points");

  DeclareCollection(m);
  DeclareLazyArray<int>(m, "LazyArrayInt");
  DeclarePolygonCollection(m);
  DeclareRegion(m);

  // Marching squares
  m.def("find_contours", &FindContoursPython, py::arg("image"),
        py::arg("level") = 0.5,
        R"pbdoc(
        Find iso-valued contours in a binary 2D array using marching squares.

        Uses the marching squares algorithm to compute contours at the specified
        level. Array values are linearly interpolated to provide better precision.
        Contours that touch the image border are automatically closed by adding
        corner points along the border.

        Contours wind counter-clockwise around low-valued regions. For binary
        masks (0/1), uint8 is the most efficient dtype. NaN values in float
        arrays are skipped. Uses low-value connectivity for ambiguous cases.

        This implementation is based on scikit-image's marching squares algorithm
        (BSD-3-Clause license).

        Args:
            image (ndarray): Input binary image of shape (M, N) in which to find
                contours. Must be a 2D numpy array. Accepts uint8, int32, float32,
                or float64 types. For binary masks, uint8 is recommended.
            level (float, optional): The iso-value level at which to extract contours.
                Default is 0.5. For binary masks, use 0.5 for centered contours, or
                values like 0.1 or 0.9 for pixel-aligned boundaries that avoid
                half-pixel offsets.

        Returns:
            list[Polygon]: List of Polygon objects representing the contours.
                Each Polygon has exterior coordinates tracing the contour path.

        Raises:
            ValueError: If input array is not 2-dimensional, has invalid dimensions
                (< 2x2), or has an unsupported dtype.

        Example:
            >>> import numpy as np
            >>> from dlup.geometry import Polygon
            >>> mask = np.zeros((5, 5), dtype=np.uint8)
            >>> mask[1:4, 1:4] = 1
            >>> contours = dlup._geometry.find_contours(mask, level=0.9)
            >>> len(contours)
            1
            >>> type(contours[0])
            <class 'dlup.geometry.Polygon'>
        )pbdoc");

  py::register_exception<dlup::geometry::GeometryError>(m, "GeometryError");
  py::register_exception<dlup::geometry::GeometryIntersectionError>(
      m, "GeometryIntersectionError");
  py::register_exception<dlup::geometry::GeometryTransformationError>(
      m, "GeometryTransformationError");
  py::register_exception<dlup::geometry::GeometryFactoryFunctionError>(
      m, "GeometryFactoryFunctionError");
  py::register_exception<dlup::geometry::GeometryNotFoundError>(
      m, "GeometryNotFoundError");
  py::register_exception<dlup::geometry::GeometryCoordinatesError>(
      m, "GeometryCoordinatesError");
}
