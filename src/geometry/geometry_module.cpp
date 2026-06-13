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
#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "aifocore/math/ndarray.h"
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

namespace nb = nanobind;

template class FactoryManager<dlup::geometry::Polygon>;
template class FactoryManager<dlup::geometry::Box>;
template class FactoryManager<dlup::geometry::Point>;

inline void DeclarePoint(nb::module_& m) {
  nb::class_<dlup::geometry::Point, dlup::geometry::BaseGeometry>(m, "Point")
      .def(nb::init<>())
      .def(nb::init<const dlup::geometry::BoostPoint&>())
      .def(nb::init<double, double>())
      .def("__init__",
           [](dlup::geometry::Point* self,
              const std::shared_ptr<dlup::geometry::Point>& p) {
             // Copy parameters from the existing object so the wrapping Python
             // class observes the same fields.
             new (self) dlup::geometry::Point(*p->point_);
             self->parameters_ = p->parameters_;
           })
      .def("__init__",
           [](dlup::geometry::Point* self, const dlup::geometry::Point& other) {
             new (self) dlup::geometry::Point(*other.point_);
             self->parameters_ = other.parameters_;
           })
      .def_prop_ro("coordinates", &dlup::geometry::Point::GetCoordinates,
                   "Get the coordinates of the point as an (x, y) tuple")
      .def_prop_ro("x", &dlup::geometry::Point::GetX, "Get the X coordinate")
      .def_prop_ro("y", &dlup::geometry::Point::GetY, "Get the Y coordinate")
      .def("distance_to", &dlup::geometry::Point::DistanceTo, nb::arg("other"),
           "Calculate the distance to another point")
      .def("equals", &dlup::geometry::Point::Equals, nb::arg("other"),
           "Check if the point is equal to another point")
      .def("within", &dlup::geometry::Point::Within, nb::arg("polygon"),
           "Check if the point is within a polygon")
      .def("scale", &dlup::geometry::Point::Scale, nb::arg("scaling"),
           "Scale the point in-place point by a factor")
      .def_prop_ro("wkt", &dlup::geometry::Point::ToWkt,
                   "Get the WKT representation of the point");
}

inline void DeclarePolygon(nb::module_& m) {
  nb::class_<dlup::geometry::Polygon, dlup::geometry::BaseGeometry>(m,
                                                                    "Polygon")
      .def(nb::init<>())
      .def(nb::init<const dlup::geometry::BoostPolygon&>())
      .def(nb::init<
           const std::vector<std::pair<double, double>>&,
           const std::vector<std::vector<std::pair<double, double>>>&>())
      .def("__init__",
           [](dlup::geometry::Polygon* self,
              const std::shared_ptr<dlup::geometry::Polygon>& p) {
             new (self) dlup::geometry::Polygon(*p->polygon_);
             self->parameters_ = p->parameters_;
           })
      .def("__init__",
           [](dlup::geometry::Polygon* self,
              const dlup::geometry::Polygon& other) {
             new (self) dlup::geometry::Polygon(*other.polygon_);
             self->parameters_ = other.parameters_;
           })
      .def("set_exterior", &dlup::geometry::Polygon::SetExterior)
      .def("set_interiors", &dlup::geometry::Polygon::SetInteriors)
      .def("get_exterior", &dlup::geometry::Polygon::GetExterior)
      .def("get_exterior_iterator",
           [](dlup::geometry::Polygon& self) {
             return nb::make_iterator(nb::type<dlup::geometry::Polygon>(),
                                      "exterior_iterator",
                                      self.GetExteriorAsIterator().begin(),
                                      self.GetExteriorAsIterator().end());
           })
      .def("get_interiors_iterator",
           [](dlup::geometry::Polygon& self) {
             return nb::make_iterator(nb::type<dlup::geometry::Polygon>(),
                                      "interiors_iterator",
                                      self.GetInteriorAsIterator().begin(),
                                      self.GetInteriorAsIterator().end());
           })
      .def("scale", &dlup::geometry::Polygon::Scale, nb::arg("scaling"))
      .def("get_interiors", &dlup::geometry::Polygon::GetInteriors)
      .def("correct_orientation", &dlup::geometry::Polygon::CorrectIfNeeded)
      .def("simplify", &dlup::geometry::Polygon::SimplifyPolygon)
      .def("contains", &dlup::geometry::Polygon::Contains, nb::arg("other"),
           "Check if the polygon fully contains another polygon. Does not "
           "check if the fields are equal")
      .def("make_valid", &dlup::geometry::Polygon::MakeValid,
           "Make the polygon valid by removing self-intersections and "
           "duplicate points")
      .def("equals", &dlup::geometry::Polygon::Equals, nb::arg("other"),
           "Check if the polygon is equal to another polygon. Checks if the "
           "fields are equal.")
      .def_prop_ro("wkt", &dlup::geometry::Polygon::ToWkt)
      .def_prop_ro("is_valid", &dlup::geometry::Polygon::IsValid)
      .def_prop_ro("area", &dlup::geometry::Polygon::GetArea)
      .def_prop_ro("bounding_box", &dlup::geometry::Polygon::GetBoundingBox);
}

void DeclareCollection(nb::module_& m) {
  using GeometryCollection = dlup::geometry::GeometryCollection;
  using PolygonPtr = std::shared_ptr<dlup::geometry::Polygon>;
  using BoxPtr = std::shared_ptr<dlup::geometry::Box>;
  using PointPtr = std::shared_ptr<dlup::geometry::Point>;

  nb::class_<GeometryCollection>(m, "GeometryCollection")
      .def(nb::init<>())
      .def("__getstate__",
           [](const GeometryCollection& collection) {
             nb::dict state;
             state["polygons"] = collection.GetPolygons();
             state["points"] = collection.GetPoints();
             state["boxes"] = collection.GetBoxes();
             state["rois"] = collection.GetRois();
             state["rtree_invalidated"] = collection.IsRTreeInvalidated();
             return state;
           })
      .def("__setstate__",
           [](GeometryCollection& self, const nb::dict& state) {
             new (&self) GeometryCollection();

             const bool was_rtree_invalidated =
                 nb::cast<bool>(state["rtree_invalidated"]);

             for (const auto& polygon :
                  nb::cast<std::vector<PolygonPtr>>(state["polygons"])) {
               self.AddPolygon(polygon);
             }
             for (const auto& point :
                  nb::cast<std::vector<PointPtr>>(state["points"])) {
               self.AddPoint(point);
             }
             for (const auto& box :
                  nb::cast<std::vector<BoxPtr>>(state["boxes"])) {
               self.AddBox(box);
             }
             for (const auto& roi :
                  nb::cast<std::vector<PolygonPtr>>(state["rois"])) {
               self.AddRoi(roi);
             }

             if (!was_rtree_invalidated) {
               self.RebuildRTree();
             }
           })

      .def("add_polygon", &GeometryCollection::AddPolygon)
      .def("add_roi", &GeometryCollection::AddRoi)
      .def("add_point", &GeometryCollection::AddPoint)
      .def("add_box", &GeometryCollection::AddBox)
      .def_prop_ro("num_polygons", &GeometryCollection::NumPolygons)
      .def_prop_ro("num_rois", &GeometryCollection::NumRois)
      .def_prop_ro("num_points", &GeometryCollection::NumPoints)
      .def_prop_ro("num_boxes", &GeometryCollection::NumBoxes)
      .def_prop_ro("has_rois", &GeometryCollection::HasRois)

      // Overloads dispatched explicitly via member function pointer casts since
      // nanobind has no overload_cast helper.
      .def("remove_polygon",
           static_cast<void (GeometryCollection::*)(const PolygonPtr&)>(
               &GeometryCollection::RemovePolygon),
           "Remove a polygon by passing the Polygon object")
      .def("remove_polygon",
           static_cast<void (GeometryCollection::*)(size_t)>(
               &GeometryCollection::RemovePolygon),
           "Remove a polygon by its index")
      .def("remove_box",
           static_cast<void (GeometryCollection::*)(const BoxPtr&)>(
               &GeometryCollection::RemoveBox),
           "Remove a box by passing the Box object")
      .def("remove_box",
           static_cast<void (GeometryCollection::*)(size_t)>(
               &GeometryCollection::RemoveBox),
           "Remove a box by its index")
      .def("remove_roi",
           static_cast<void (GeometryCollection::*)(const PolygonPtr&)>(
               &GeometryCollection::RemoveRoi),
           "Remove an ROI by passing the ROI object")
      .def("remove_roi",
           static_cast<void (GeometryCollection::*)(size_t)>(
               &GeometryCollection::RemoveRoi),
           "Remove an ROI by its index")
      .def("reindex_polygons", &GeometryCollection::ReindexPolygons)
      .def(
          "sort_polygons",
          [](GeometryCollection& self, const nb::callable& key_func,
             bool reverse) {
            self.SortPolygons([&key_func, reverse](const auto& a,
                                                   const auto& b) {
              nb::object key_a = key_func(a);
              nb::object key_b = key_func(b);

              if (nb::isinstance<nb::str>(key_a) &&
                  nb::isinstance<nb::str>(key_b)) {
                return reverse ? (nb::cast<std::string>(key_a) >
                                  nb::cast<std::string>(key_b))
                               : (nb::cast<std::string>(key_a) <
                                  nb::cast<std::string>(key_b));
              } else if (nb::isinstance<nb::float_>(key_a) &&
                         nb::isinstance<nb::float_>(key_b)) {
                return reverse
                           ? (nb::cast<double>(key_a) > nb::cast<double>(key_b))
                           : (nb::cast<double>(key_a) <
                              nb::cast<double>(key_b));
              } else if (nb::isinstance<nb::int_>(key_a) &&
                         nb::isinstance<nb::int_>(key_b)) {
                return reverse ? (nb::cast<int>(key_a) > nb::cast<int>(key_b))
                               : (nb::cast<int>(key_a) < nb::cast<int>(key_b));
              } else if (key_a.is_none() && key_b.is_none()) {
                return false;
              } else {
                throw std::invalid_argument(
                    "Unsupported key type for sorting.");
              }
            });
          },
          "Sort polygons by a custom key function")
      .def("simplify_polygons", &GeometryCollection::SimplifyPolygons)
      .def("__len__", &GeometryCollection::Size)

      .def("remove_point",
           static_cast<void (GeometryCollection::*)(const PointPtr&)>(
               &GeometryCollection::RemovePoint),
           "Remove a point by passing the Point object")
      .def("remove_point",
           static_cast<void (GeometryCollection::*)(size_t)>(
               &GeometryCollection::RemovePoint),
           "Remove a point by its index")
      .def("read_region", &GeometryCollection::ReadRegion)
      .def("rebuild_rtree", &GeometryCollection::RebuildRTree,
           "Rebuild the R-tree index manually")
      .def("scale", &GeometryCollection::Scale,
           "Scale all geometries by a factor")
      .def("set_offset", &GeometryCollection::SetOffset,
           "Set an offset for all geometries")
      .def_prop_ro("rtree_invalidated", &GeometryCollection::IsRTreeInvalidated)
      .def_prop_ro("pointer_id", &GeometryCollection::GetPointerId)
      .def_prop_ro("bounding_box", &GeometryCollection::ComputeBoundingBox)
      .def_prop_ro(
          "polygons",
          [](GeometryCollection& self) {
            nb::list py_polygons;
            for (const auto& polygon : self.GetPolygons()) {
              nb::object processed_polygon =
                  FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                      polygon);
              py_polygons.append(processed_polygon);
            }
            return py_polygons;
          })
      .def_prop_ro(
          "rois",
          [](GeometryCollection& self) {
            nb::list py_rois;
            for (const auto& roi : self.GetRois()) {
              nb::object processed_roi =
                  FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                      roi);
              py_rois.append(processed_roi);
            }
            return py_rois;
          })
      .def_prop_ro(
          "points",
          [](GeometryCollection& self) {
            nb::list py_points;
            for (const auto& point : self.GetPoints()) {
              nb::object processed_point =
                  FactoryManager<dlup::geometry::Point>::CallFactoryFunction(
                      point);
              py_points.append(processed_point);
            }
            return py_points;
          })
      .def_prop_ro(
          "boxes",
          [](GeometryCollection& self) {
            nb::list py_boxes;
            for (const auto& box : self.GetBoxes()) {
              nb::object processed_box =
                  FactoryManager<dlup::geometry::Box>::CallFactoryFunction(box);
              py_boxes.append(processed_box);
            }
            return py_boxes;
          })
      .def_prop_ro("index_map", &GeometryCollection::GetIndexMap);
}

void DeclarePolygonCollection(nb::module_& m) {
  nb::class_<dlup::geometry::PolygonCollection>(m, "PolygonCollection")
      .def("get_geometries",
           [](dlup::geometry::PolygonCollection& self) {
             nb::list py_polygons;
             for (const auto& polygon : self.GetGeometries()) {
               nb::object processed_polygon =
                   FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                       polygon);
               py_polygons.append(processed_polygon);
             }
             return py_polygons;
           })
      .def("to_mask", &dlup::geometry::PolygonCollection::ToMask,
           nb::arg("default_value") = 0);
}

inline void DeclareBaseGeometry(nb::module_& m) {
  nb::class_<dlup::geometry::BaseGeometry>(m, "BaseGeometry")
      .def(
          "set_field",
          [](dlup::geometry::BaseGeometry& self, const std::string& name,
             nb::object value) {
            // ``None`` clears the slot via the ``std::monostate`` alternative.
            if (value.is_none()) {
              self.SetField(name, std::monostate{});
              return;
            }
            FieldType field_value;
            if (!nb::try_cast<FieldType>(value, field_value)) {
              throw nb::type_error(
                  "set_field: value type is not supported by FieldType");
            }
            self.SetField(name, std::move(field_value));
          },
          nb::arg("name"), nb::arg("value").none())
      .def(
          "get_field",
          [](dlup::geometry::BaseGeometry& self,
             const std::string& name) -> nb::object {
            auto field = self.GetField(name);
            if (!field) {
              return nb::none();
            }
            return std::visit(
                [](const auto& value) -> nb::object { return nb::cast(value); },
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
      .def_prop_ro("fields", &dlup::geometry::BaseGeometry::GetFields)
      .def_prop_ro("pointer_id", &dlup::geometry::BaseGeometry::GetPointerId);
}

inline void DeclareBox(nb::module_& m) {
  nb::class_<dlup::geometry::Box, dlup::geometry::BaseGeometry>(m, "Box")
      .def(nb::init<>())
      .def(nb::init<const dlup::geometry::BoostBox&>())
      .def(nb::init<const std::array<double, 2>&,
                    const std::array<double, 2>&>())
      .def("__init__",
           [](dlup::geometry::Box* self,
              const std::shared_ptr<dlup::geometry::Box>& p) {
             new (self) dlup::geometry::Box(*p->box_);
             self->parameters_ = p->parameters_;
           })
      .def("__init__",
           [](dlup::geometry::Box* self, const dlup::geometry::Box& other) {
             new (self) dlup::geometry::Box(*other.box_);
             self->parameters_ = other.parameters_;
           })
      .def(
          "as_polygon",
          [](const dlup::geometry::Box& box) {
            auto polygon = box.AsPolygon();
            return FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(
                polygon);
          },
          "Convert the box to a polygon")
      .def("scale", &dlup::geometry::Box::Scale, nb::arg("scaling"),
           "Scale the box in-place by a factor")
      .def_prop_ro(
          "coordinates",
          [](const dlup::geometry::Box& self) {
            auto coords = self.GetCoordinates();
            return nb::make_tuple(coords[0], coords[1]);
          },
          "Get the top-left coordinates of the box as an (x, y) tuple")
      .def_prop_ro(
          "size",
          [](const dlup::geometry::Box& self) {
            auto size = self.GetSize();
            return nb::make_tuple(size[0], size[1]);
          },
          "Get the size of the box as an (h, w) tuple")
      .def_prop_ro("area", &dlup::geometry::Box::GetArea)
      .def_prop_ro("wkt", &dlup::geometry::Box::ToWkt,
                   "Get the WKT representation of the box");
}

template <typename T>
void DeclareLazyArray(nb::module_& m, const std::string& type_name) {
  nb::class_<LazyArray<T>>(m, type_name.c_str())
      .def(nb::init<typename LazyArray<T>::ComputeFunction,
                    std::vector<std::size_t>>())
      .def("numpy",
           [](const LazyArray<T>& arr) {
             const auto& data = arr.data();
             const auto& shape = arr.shape();

             // Allocate a buffer that the returned numpy array will own via a
             // capsule. We avoid sharing storage with the LazyArray's internal
             // vector because the latter is `mutable` and may be invalidated
             // by future calls; the capsule keeps the lifetime independent.
             auto* buffer = new T[data.size()];
             std::copy(data.begin(), data.end(), buffer);

             nb::capsule owner(buffer, [](void* ptr) noexcept {
               delete[] static_cast<T*>(ptr);
             });

             // Cast shape (size_t) into the size type expected by ndarray
             // (size_t already matches the constructor signature).
             return nb::ndarray<nb::numpy, T>(buffer, shape.size(),
                                              shape.data(), owner);
           })
      .def("shape", &LazyArray<T>::shape)
      .def("__repr__", [](const LazyArray<T>&) {
        return "<LazyArray: use numpy() to compute>";
      });
}

void DeclareRegion(nb::module_& m) {
  nb::class_<dlup::geometry::AnnotationRegion>(m, "AnnotationRegion")
      .def(nb::init<std::function<dlup::geometry::AnnotationRegion()>, bool>())
      .def(nb::init<std::vector<std::shared_ptr<dlup::geometry::Polygon>>,
                    std::vector<std::shared_ptr<dlup::geometry::Polygon>>,
                    std::vector<std::shared_ptr<dlup::geometry::Box>>,
                    std::vector<std::shared_ptr<dlup::geometry::Point>>,
                    std::tuple<int, int>, bool>())
      .def_prop_ro("polygons", &dlup::geometry::AnnotationRegion::GetPolygons)
      .def_prop_ro("rois", &dlup::geometry::AnnotationRegion::GetRois)
      .def_prop_ro(
          "boxes",
          [](dlup::geometry::AnnotationRegion& self) {
            auto boxes = self.GetBoxes();
            nb::list py_boxes;
            for (const auto& box : boxes) {
              py_boxes.append(
                  FactoryManager<dlup::geometry::Box>::CallFactoryFunction(
                      box));
            }
            return py_boxes;
          })
      .def_prop_ro(
          "points",
          [](dlup::geometry::AnnotationRegion& self) {
            auto points = self.GetPoints();
            nb::list py_points;
            for (const auto& point : points) {
              py_points.append(
                  FactoryManager<dlup::geometry::Point>::CallFactoryFunction(
                      point));
            }
            return py_points;
          })
      .def_prop_ro("has_rois", &dlup::geometry::AnnotationRegion::HasRois);
}

namespace {

// Accepts a 2D contiguous CPU view of any supported dtype and dispatches to
// FindContours via an aifocore NDArrayView. The view is non-owning; only the
// non-double conversion path materialises a temporary buffer.
template <typename T>
std::vector<std::shared_ptr<dlup::geometry::Polygon>> ProcessArrayForContours(
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu> image,
    double level) {
  const std::size_t height = image.shape(0);
  const std::size_t width = image.shape(1);

  if constexpr (std::is_same_v<T, double>) {
    aifocore::math::NDArrayView<double, 2> ndarray_view(
        const_cast<double*>(image.data()), {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  } else {
    std::vector<double> converted_data(height * width);
    const T* src = image.data();
    for (std::size_t i = 0; i < height * width; ++i) {
      converted_data[i] = static_cast<double>(src[i]);
    }

    aifocore::math::NDArrayView<double, 2> ndarray_view(converted_data.data(),
                                                        {height, width});
    return dlup::geometry::FindContours(ndarray_view, level);
  }
}

nb::list FindContoursPython(const nb::object& image, double level) {
  std::vector<std::shared_ptr<dlup::geometry::Polygon>> contours;

  // Use untyped ndarray to inspect the dtype, then forward to a typed view.
  using AnyArray =
      nb::ndarray<nb::ro, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  AnyArray any_array;
  try {
    any_array = nb::cast<AnyArray>(image);
  } catch (const std::exception&) {
    throw std::invalid_argument(
        "Input array must be 2-dimensional and C-contiguous on the CPU");
  }

  const auto dtype = any_array.dtype();
  if (dtype == nb::dtype<bool>()) {
    contours = ProcessArrayForContours<bool>(
        nb::cast<nb::ndarray<const bool, nb::ndim<2>, nb::c_contig,
                             nb::device::cpu>>(image),
        level);
  } else if (dtype == nb::dtype<uint8_t>()) {
    contours = ProcessArrayForContours<uint8_t>(
        nb::cast<nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig,
                             nb::device::cpu>>(image),
        level);
  } else if (dtype == nb::dtype<int32_t>()) {
    contours = ProcessArrayForContours<int32_t>(
        nb::cast<nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig,
                             nb::device::cpu>>(image),
        level);
  } else if (dtype == nb::dtype<float>()) {
    contours = ProcessArrayForContours<float>(
        nb::cast<nb::ndarray<const float, nb::ndim<2>, nb::c_contig,
                             nb::device::cpu>>(image),
        level);
  } else if (dtype == nb::dtype<double>()) {
    contours = ProcessArrayForContours<double>(
        nb::cast<nb::ndarray<const double, nb::ndim<2>, nb::c_contig,
                             nb::device::cpu>>(image),
        level);
  } else {
    throw std::invalid_argument(
        "Input array must be of type bool, uint8, int32, float32, or float64");
  }

  nb::list result;
  for (const auto& polygon : contours) {
    nb::object py_polygon =
        FactoryManager<dlup::geometry::Polygon>::CallFactoryFunction(polygon);
    result.append(py_polygon);
  }

  return result;
}

}  // namespace

NB_MODULE(_geometry, m) {
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

  m.def("find_contours", &FindContoursPython, nb::arg("image"),
        nb::arg("level") = 0.5,
        R"pbdoc(
        Find iso-valued contours in a binary 2D array using marching squares.

        Uses the marching squares algorithm to compute contours at the specified
        level. Array values are linearly interpolated to provide better precision.
        Contours that touch the image border are automatically closed by adding
        corner points along the border.

        Contours wind counter-clockwise around low-valued regions. For binary
        masks (0/1), uint8 is the most efficient dtype. NaN values in float
        arrays are skipped. Uses low-value connectivity for ambiguous cases.

        Nested contours are combined into polygons with holes: a contour enclosed
        by another becomes an interior ring on its parent, while a contour that
        sits inside a hole is emitted as a new top-level polygon (which may
        itself contain holes). The returned list therefore contains one Polygon
        per connected high-valued region, each with its own set of holes.

        This implementation is based on scikit-image's marching squares algorithm
        (BSD-3-Clause license).

        Args:
            image (ndarray): Input binary image of shape (M, N) in which to find
                contours. Must be a 2D numpy array. Accepts bool, uint8, int32,
                float32, or float64 types. For binary masks, bool or uint8 is
                recommended.
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

  nb::exception<dlup::geometry::GeometryError>(m, "GeometryError");
  nb::exception<dlup::geometry::GeometryIntersectionError>(
      m, "GeometryIntersectionError");
  nb::exception<dlup::geometry::GeometryTransformationError>(
      m, "GeometryTransformationError");
  nb::exception<dlup::geometry::GeometryFactoryFunctionError>(
      m, "GeometryFactoryFunctionError");
  nb::exception<dlup::geometry::GeometryNotFoundError>(m,
                                                       "GeometryNotFoundError");
  nb::exception<dlup::geometry::GeometryCoordinatesError>(
      m, "GeometryCoordinatesError");
}
