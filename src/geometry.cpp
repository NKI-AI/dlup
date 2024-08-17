#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "exceptions.h"
#include "geometry.h"
#include "geometry_utils.h"
#include "opencv.h"
#include "region.h"
#include "rtree.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// #define DLUPDEBUG

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;

namespace py = pybind11;

std::vector<std::pair<double, double>> Polygon::getExterior() const {
    std::vector<std::pair<double, double>> result;
    result.reserve(bg::exterior_ring(*polygon).size());
    for (const auto &point : bg::exterior_ring(*polygon)) {
        result.emplace_back(bg::get<0>(point), bg::get<1>(point));
    }
    return result;
}

std::vector<std::vector<std::pair<double, double>>> Polygon::getInteriors() const {
    // correctIfNeeded();
    std::vector<std::vector<std::pair<double, double>>> result;
    result.reserve(polygon->inners().size());
    for (const auto &inner : polygon->inners()) {
        std::vector<std::pair<double, double>> inner_result;
        for (const auto &point : inner) {
            inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
        }
        result.emplace_back(inner_result);
    }
    return result;
}

void Polygon::correctIfNeeded() const {
    if (!isCorrected) {
        bg::correct(*polygon); // Dereference the shared pointer to apply the correction
        isCorrected = true;
    }
}

void Polygon::setExterior(const std::vector<std::pair<double, double>> &coordinates) {
    bg::exterior_ring(*polygon).clear();
    bg::exterior_ring(*polygon).reserve(coordinates.size());
    for (const auto &coord : coordinates) {
        bg::append(*polygon, BoostPoint(coord.first, coord.second));
    }

    // Close the ring if it's not already closed
    // Shapely does this, so we want to keep compatibility.
    if (coordinates.front() != coordinates.back()) {
        bg::append(*polygon, BoostPoint(coordinates.front().first, coordinates.front().second));
    }

    isCorrected = false; // Mark as not corrected. Correction reorients and closes
}

void Polygon::setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors) {
    bg::interior_rings(*polygon).clear();
    polygon->inners().resize(interiors.size());

    for (size_t i = 0; i < interiors.size(); ++i) {
        const auto &interior_coords = interiors[i];
        auto &inner = polygon->inners()[i];
        inner.clear();

        for (const auto &coord : interior_coords) {
            bg::append(inner, BoostPoint(coord.first, coord.second));
        }

        // Close the ring if it's not already closed
        if (interior_coords.front() != interior_coords.back()) {
            bg::append(inner, BoostPoint(interior_coords.front().first, interior_coords.front().second));
        }
    }

    isCorrected = false; // Mark as not corrected. Correction reorients and closes
}

void Polygon::scale(double scaling) {
    GeometryUtils::applyAffineTransformation(*polygon, {0.0, 0.0}, scaling);
}

std::vector<std::shared_ptr<Polygon>> Polygon::intersection(const BoostPolygon &otherPolygon) const {
    // correctIfNeeded();
    // Make the polygon valid if needed before performing the intersection
    // TODO: This simplifies the polygon!!
    BoostPolygon validPolygon = GeometryUtils::makeValid(*polygon);

    std::vector<BoostPolygon> intersectionResult;
    // intersectionResult.reserve(validPolygon.inners().size() * 5);
    bg::intersection(validPolygon, otherPolygon, intersectionResult);

    std::vector<std::shared_ptr<Polygon>> result;
    for (const auto &intersectedBoostPolygon : intersectionResult) {
        auto intersectedPolygon = std::make_shared<Polygon>(intersectedBoostPolygon);
        // Copy the parameters from this polygon to the new one

        for (const auto &param : parameters) {
            intersectedPolygon->setField(param.first, param.second);
        }

        result.emplace_back(intersectedPolygon);
    }

    return result;
}

void Polygon::simplifyPolygon(double tolerance) { bg::simplify(*polygon, *polygon, tolerance); }

py::list AnnotationRegion::getPolygons() const {
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif
    py::list py_polygons;
    for (const auto &polygon : polygons_) {
        py_polygons.append(callFactoryFunction(polygon));
    }
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in AnnotationRegion::getPolygons: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - end).count() << " ms" << std::endl;
#endif
    return py_polygons;
}

py::list AnnotationRegion::getPoints() const {
    py::list py_points;
    for (const auto &point : points_) {
        py_points.append(callFactoryFunction(point));
    }
    return py_points;
}

class GeometryCollection {
public:
    GeometryCollection();
    // Whatever any LLM says this has to be a shared pointer as we share it with the python interpreter
    using PolygonPtr = std::shared_ptr<Polygon>;
    using PointPtr = std::shared_ptr<Point>;

    std::vector<PolygonPtr> polygons;
    std::vector<PointPtr> points;
    RTreeWrapper rtreeWrapper;

    void addPolygon(const PolygonPtr &p);
    void addPoint(const PointPtr &p);

    py::list getPolygons();
    py::list getPoints();
    std::pair<std::pair<double, double>, std::pair<double, double>> computeBoundingBox() const;
    void sortPolygons(const py::function &keyFunc, bool reverse);

    void removePolygon(const PolygonPtr &p);
    void removePolygon(size_t index);
    void removePoint(const PointPtr &p);
    void removePoint(size_t index);

    void scale(double scaling);
    void setOffset(std::pair<double, double> offset);
    void rebuildRTree() { rtreeWrapper.rebuild(); }
    void simplifyPolygons(double tolerance) {
        for (auto &polygon : polygons) {
            polygon->simplifyPolygon(tolerance);
        }
    }

    int size() const { return polygons.size() + points.size(); }

    std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

    bool isRTreeInvalidated() const { return rtreeWrapper.isInvalidated(); }

    AnnotationRegion readRegion(const std::pair<double, double> &coordinates, double scaling,
                                const std::pair<double, double> &size);

    // TODO: Rethink the need for this function.
    void reindexPolygons(const std::map<std::string, int> &indexMap);
};

GeometryCollection::GeometryCollection() : rtreeWrapper(this) {}

std::pair<std::pair<double, double>, std::pair<double, double>> GeometryCollection::computeBoundingBox() const {
    // Initialize an empty bounding box
    BoostBox overallBoundingBox;

    bool isFirst = true;

    // Iterate over all polygons and compute their bounding boxes
    for (const auto &polygon : polygons) {
        BoostBox polygonBox;
        bg::envelope(*(polygon->polygon), polygonBox);

        if (isFirst) {
            overallBoundingBox = polygonBox;
            isFirst = false;
        } else {
            bg::expand(overallBoundingBox, polygonBox);
        }
    }

    // Iterate over all points and compute their bounding boxes
    for (const auto &point : points) {
        BoostBox pointBox(*(point->point), *(point->point));

        if (isFirst) {
            overallBoundingBox = pointBox;
            isFirst = false;
        } else {
            bg::expand(overallBoundingBox, pointBox);
        }
    }

    // Extract min and max points
    const auto &min_corner = overallBoundingBox.min_corner();
    const auto &max_corner = overallBoundingBox.max_corner();

    double min_x = bg::get<0>(min_corner);
    double min_y = bg::get<1>(min_corner);
    double max_x = bg::get<0>(max_corner);
    double max_y = bg::get<1>(max_corner);

    double width = max_x - min_x;
    double height = max_y - min_y;

    return std::make_pair(std::make_pair(min_x, min_y), std::make_pair(width, height));
}

void RTreeWrapper::rebuild() {
    clear(); // Clear the existing R-tree

    // Rebuild the tree using polygons and points from GeometryCollection
    const auto &polygons = geometryCollection->polygons;
    for (size_t i = 0; i < polygons.size(); ++i) {
        BoostBox box;
        bg::envelope(*(polygons[i]->polygon), box);
        insert(box, i);
    }

    const auto &points = geometryCollection->points;
    for (size_t i = 0; i < points.size(); ++i) {
        BoostBox box(*(points[i]->point), *(points[i]->point));
        insert(box, polygons.size() + i);
    }

    rTreeInvalidated = false;
}

void GeometryCollection::addPoint(const PointPtr &p) {
    BoostBox box(*(p->point), *(p->point));
    rtreeWrapper.insert(box, polygons.size() + points.size());
    points.emplace_back(p);
}

void GeometryCollection::addPolygon(const PolygonPtr &p) {
    // Print the parameters of the polygon being added
    BoostBox box;
    bg::envelope(*(p->polygon), box);
    rtreeWrapper.insert(box, polygons.size());
    polygons.emplace_back(p);
}

py::list GeometryCollection::getPolygons() {
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
#endif
    py::list py_polygons;
    for (const auto &polygon : polygons) {
        py_polygons.append(AnnotationRegion::callFactoryFunction(polygon));
    }
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in GeometryCollection::getPolygons: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(stop - end).count() << " ms" << std::endl;
#endif
    return py_polygons;
}

py::list GeometryCollection::getPoints() {
    py::list py_points;
    for (const auto &point : points) {
        py_points.append(AnnotationRegion::callFactoryFunction(point));
    }
    return py_points;
}

void GeometryCollection::reindexPolygons(const std::map<std::string, int> &indexMap) {
    for (auto &polygon : polygons) {
        std::optional<py::object> label_opt = polygon->getField("label");

        if (label_opt.has_value()) {
            std::string label = label_opt->cast<std::string>();
            auto it = indexMap.find(label);
            if (it != indexMap.end()) {
                polygon->setField("index", py::int_(it->second));
            } else {
                throw std::invalid_argument("Label '" + label + "' not found in indexMap");
            }
        } else {
            throw std::invalid_argument("Polygon does not have a value for the 'label' field");
        }
    }
}

void GeometryCollection::sortPolygons(const py::function &keyFunc, bool reverse) {
    std::sort(polygons.begin(), polygons.end(), [&keyFunc, reverse](const PolygonPtr &a, const PolygonPtr &b) {
        py::object keyA = keyFunc(a);
        py::object keyB = keyFunc(b);

        if (py::isinstance<py::str>(keyA) && py::isinstance<py::str>(keyB)) {
            return reverse ? (keyA.cast<std::string>() > keyB.cast<std::string>())
                           : (keyA.cast<std::string>() < keyB.cast<std::string>());
        } else if (py::isinstance<py::float_>(keyA) && py::isinstance<py::float_>(keyB)) {
            return reverse ? (keyA.cast<double>() > keyB.cast<double>()) : (keyA.cast<double>() < keyB.cast<double>());
        } else if (py::isinstance<py::int_>(keyA) && py::isinstance<py::int_>(keyB)) {
            return reverse ? (keyA.cast<int>() > keyB.cast<int>()) : (keyA.cast<int>() < keyB.cast<int>());
        } else {
            throw std::invalid_argument("Unsupported key type for sorting.");
        }
    });
    rtreeWrapper.invalidate();
}

void GeometryCollection::scale(double scaling) {
    for (auto &point : points) {
        point->scale(scaling);
    }
    for (auto &polygon : polygons) {
        polygon->scale(scaling);
    }
    rtreeWrapper.invalidate();
}

void GeometryCollection::setOffset(std::pair<double, double> offset) {
    for (auto &point : points) {
        GeometryUtils::applyAffineTransformation(*point->point, {-offset.first, -offset.second}, 1.0);
    }
    for (auto &polygon : polygons) {
        GeometryUtils::applyAffineTransformation(*polygon->polygon, {-offset.first, -offset.second}, 1.0);
    }
    rtreeWrapper.invalidate();
}

void GeometryCollection::removePolygon(const PolygonPtr &p) {
    auto it = std::find(polygons.begin(), polygons.end(), p);
    if (it != polygons.end()) {
        polygons.erase(it);
        rtreeWrapper.invalidate();
    } else {
        throw GeometryNotFoundError("Polygon not found");
    }
}

void GeometryCollection::removePolygon(size_t index) {
    if (index >= polygons.size()) {
        throw std::out_of_range("Polygon index out of range");
    }

    polygons.erase(polygons.begin() + index);
    rtreeWrapper.invalidate();
}

void GeometryCollection::removePoint(const PointPtr &p) {
    auto it = std::find(points.begin(), points.end(), p);
    if (it != points.end()) {
        points.erase(it);
        rtreeWrapper.invalidate();
    } else {
        throw GeometryNotFoundError("Point not found");
    }
}

void GeometryCollection::removePoint(size_t index) {
    if (index >= points.size()) {
        throw std::out_of_range("Point index out of range");
    }

    points.erase(points.begin() + index);
    rtreeWrapper.invalidate();
}

AnnotationRegion GeometryCollection::readRegion(const std::pair<double, double> &coordinates, double scaling,
                                                const std::pair<double, double> &size) {

#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    BoostPoint topLeft(coordinates.first / scaling, coordinates.second / scaling);
    BoostPoint bottomRight((coordinates.first + size.first) / scaling, (coordinates.second + size.second) / scaling);
    BoostBox queryBox(topLeft, bottomRight);

    BoostPolygon intersectionPolygon;
    bg::convert(queryBox, intersectionPolygon);
    std::vector<std::pair<BoostBox, size_t>> results;
    rtreeWrapper.query(bgi::intersects(queryBox), std::back_inserter(results));

    std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

    // const size_t estimatedSize = 10000; //  Estimated size

    std::vector<std::shared_ptr<Polygon>> intersectedPolygons;
    std::vector<std::shared_ptr<Point>> intersectedPoints;

    // intersectedPolygons.reserve(estimatedSize);
    // intersectedPoints.reserve(estimatedSize);

    for (const auto &result : results) {
        size_t index = result.second;
        if (index < polygons.size()) {
            auto &polygon = polygons[index];
            auto intersections = polygon->intersection(intersectionPolygon);
            for (const auto &intersectedPolygon : intersections) {
                GeometryUtils::applyAffineTransformation(*intersectedPolygon->polygon, coordinates, scaling);
                intersectedPolygons.push_back(intersectedPolygon);
            }
        } else {
            auto &point = points[index - polygons.size()];
            auto transformedPoint = std::make_shared<Point>(*point);
            GeometryUtils::applyAffineTransformation(*transformedPoint->point, coordinates, scaling);
            intersectedPoints.push_back(transformedPoint);
        }
    }
#ifdef DLUPDEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time in GeometryCollection:readRegion: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
#endif
    auto returnValue = AnnotationRegion(std::move(intersectedPolygons), std::move(intersectedPoints), std::move(size));

    return returnValue;
}

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
            auto newPolygon = std::make_shared<Polygon>(*other.polygon);
            newPolygon->parameters = other.parameters; // Copy the parameters
            return newPolygon;
        }))
        .def("set_exterior", &Polygon::setExterior)
        .def("set_interiors", &Polygon::setInteriors)
        .def("get_exterior", &Polygon::getExterior)
        .def("get_exterior_iterator", [](Polygon& self) {
            return py::make_iterator(self.getExteriorAsIterator().begin(), self.getExteriorAsIterator().end());
        })
        .def("get_interiors_iterator", [](Polygon& self) {
            return py::make_iterator(self.getInteriorAsIterator().begin(), self.getInteriorAsIterator().end());
        })
        .def("scale", &Polygon::scale, py::arg("scaling"))
        .def("get_interiors", &Polygon::getInteriors)
        .def("correct_orientation", &Polygon::correctIfNeeded)
        .def("simplify", &Polygon::simplifyPolygon)
        .def_property_readonly("wkt", &Polygon::toWkt)
        .def_property_readonly("area", &Polygon::getArea);

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
            auto newPoint = std::make_shared<Point>(*other.point);
            newPoint->parameters = other.parameters; // Copy the parameters
            return newPoint;
        }))
        .def("set_coordinates", &Point::setCoordinates)
        .def("get_coordinates", &Point::getCoordinates)
        .def_property_readonly("x", &Point::getX)
        .def_property_readonly("y", &Point::getY)
        .def("distance_to", &Point::distanceTo)
        .def("equals", &Point::equals)
        .def("within", &Point::within)
        .def("centroid", &Point::centroid)
        .def("scale", &Point::scale, py::arg("scaling"))
        .def_property_readonly("wkt", &Point::toWkt);

    m.def("set_polygon_factory", &AnnotationRegion::setPolygonFactory);
    m.def("set_point_factory", &AnnotationRegion::setPointFactory);

    py::class_<GeometryCollection, std::shared_ptr<GeometryCollection>>(m, "GeometryCollection")
        .def(py::init<>())
        .def("add_polygon", &GeometryCollection::addPolygon)
        .def("add_point", &GeometryCollection::addPoint)

        // Overload remove_polygon to handle both object and index
        .def("remove_polygon", py::overload_cast<const std::shared_ptr<Polygon> &>(&GeometryCollection::removePolygon),
             "Remove a polygon by passing the Polygon object")
        .def("remove_polygon", py::overload_cast<size_t>(&GeometryCollection::removePolygon),
             "Remove a polygon by its index")
        .def("reindex_polygons", &GeometryCollection::reindexPolygons)
        .def("sort_polygons", &GeometryCollection::sortPolygons, "Sort polygons by a custom key function")
        .def("simplify_polygons", &GeometryCollection::simplifyPolygons)
        .def("size", &GeometryCollection::size)

        // Overload remove_point to handle both object and index
        .def("remove_point", py::overload_cast<const std::shared_ptr<Point> &>(&GeometryCollection::removePoint),
             "Remove a point by passing the Point object")
        .def("remove_point", py::overload_cast<size_t>(&GeometryCollection::removePoint), "Remove a point by its index")
        .def("read_region", &GeometryCollection::readRegion)
        .def("rebuild_rtree", &GeometryCollection::rebuildRTree, "Rebuild the R-tree index manually")
        .def("scale", &GeometryCollection::scale, "Scale all geometries by a factor")
        .def("set_offset", &GeometryCollection::setOffset, "Set an offset for all geometries")
        .def_property_readonly("rtree_invalidated", &GeometryCollection::isRTreeInvalidated)
        .def_property_readonly("pointer_id", &GeometryCollection::getPointerId)
        .def_property_readonly("bounding_box", &GeometryCollection::computeBoundingBox)
        .def_property_readonly("polygons", &GeometryCollection::getPolygons)
        .def_property_readonly("points", &GeometryCollection::getPoints);

    py::class_<AnnotationRegion, std::shared_ptr<AnnotationRegion>>(m, "AnnotationRegion")
        .def_property_readonly("polygons", &AnnotationRegion::getPolygons)
        .def_property_readonly("points", &AnnotationRegion::getPoints)
        .def("to_mask", &AnnotationRegion::toMask, py::arg("default_value") = 0);

    py::register_exception<GeometryError>(m, "GeometryError");
    py::register_exception<GeometryIntersectionError>(m, "GeometryIntersectionError");
    py::register_exception<GeometryTransformationError>(m, "GeometryTransformationError");
    py::register_exception<GeometryFactoryFunctionError>(m, "GeometryFactoryFunctionError");
    py::register_exception<GeometryNotFoundError>(m, "GeometryNotFoundError");
    py::register_exception<GeometryCoordinatesError>(m, "GeometryCoordinatesError");
}