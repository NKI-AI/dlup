#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include "exceptions.h"
#include "geometry.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;
using BoostLineString = bg::model::linestring<BoostPoint>;
using BoostMultiPolygon = bg::model::multi_polygon<BoostPolygon>;


class FactoryGuard {
public:
    FactoryGuard(py::function& factory_ref, py::function new_factory)
        : factory_ref_(factory_ref), original_factory_(factory_ref) {
        factory_ref_ = new_factory;
    }

    ~FactoryGuard() {
        factory_ref_ = original_factory_;
    }

private:
    py::function& factory_ref_;
    py::function original_factory_;
};

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void setField(const std::string &name, py::object value) { parameters[name] = value; }

    std::optional<py::object> getField(const std::string &name) const {
        auto it = parameters.find(name);
        if (it != parameters.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<std::string> getFields() const {
        std::vector<std::string> fieldNames;
        fieldNames.reserve(parameters.size());
        std::transform(parameters.begin(), parameters.end(), std::back_inserter(fieldNames),
                       [](const auto &param) { return param.first; });
        return fieldNames;
    }

    std::uintptr_t getPointerId() const { return reinterpret_cast<std::uintptr_t>(this); }

    virtual std::string toWkt() const = 0; // Force derived classes to provide the WKT

protected:
    template <typename GeometryType>
    std::string convertToWkt(const GeometryType &geometry) const {
        std::stringstream ss;
        ss << boost::geometry::wkt(geometry);
        return ss.str();
    }
};

class Polygon : public BaseGeometry {
public:
    ~Polygon() override = default;
    std::shared_ptr<BoostPolygon> polygon;

    Polygon() : polygon(std::make_shared<BoostPolygon>()) {}
    Polygon(const BoostPolygon &p) : polygon(std::make_shared<BoostPolygon>(p)) {}
    Polygon(std::shared_ptr<BoostPolygon> p) : polygon(p) {}

    Polygon(const std::vector<std::pair<double, double>> &exterior,
            const std::vector<std::vector<std::pair<double, double>>> &interiors = {})
        : polygon(std::make_shared<BoostPolygon>()) {
        setExterior(std::move(exterior));
        setInteriors(std::move(interiors));
    }

    // TODO: We don't just need to intersect with a box, but with any geometry
    // TODO: Need to remark that it only intersects with boost-type structures
    // TODO: If we extend it we don't want conflicts between parameters
    // std::vector<std::shared_ptr<Polygon>> intersection(const BoostPolygon &otherPolygon) const {
    //     std::vector<BoostPolygon> intersectionResult;
    //     bg::intersection(*polygon, otherPolygon, intersectionResult);

    //     std::vector<std::shared_ptr<Polygon>> result;
    //     for (const auto &intersectedBoostPolygon : intersectionResult) {
    //         auto intersectedPolygon = std::make_shared<Polygon>(intersectedBoostPolygon);
    //         // Copy the parameters from this polygon to the new one
    //         for (const auto &param : parameters) {
    //             intersectedPolygon->setField(param.first, param.second);
    //         }

    //         result.push_back(intersectedPolygon);
    //     }

    //     return result;
    // }
    std::vector<std::shared_ptr<Polygon>> intersection(const BoostPolygon &otherPolygon) const {
        // Make the polygon valid before performing the intersection
        BoostPolygon validPolygon = GeometryUtils::makeValid(*polygon);

        std::vector<BoostPolygon> intersectionResult;
        bg::intersection(validPolygon, otherPolygon, intersectionResult);

        std::vector<std::shared_ptr<Polygon>> result;
        for (const auto &intersectedBoostPolygon : intersectionResult) {
            auto intersectedPolygon = std::make_shared<Polygon>(intersectedBoostPolygon);
            // Copy the parameters from this polygon to the new one
            for (const auto &param : parameters) {
                intersectedPolygon->setField(param.first, param.second);
            }

            result.push_back(intersectedPolygon);
        }

        return result;
    }

    // This does a smart merge, but seems to impact performance quite significantly.
    // We need to check downstream, e.g. after creating a mask if the unionizing is worth doing.
    // std::vector<std::shared_ptr<Polygon>> intersection(const BoostBox &box) const {
    //     std::vector<BoostPolygon> intersectionResult;
    //     bg::intersection(*polygon, box, intersectionResult);

    //     if (intersectionResult.empty()) {
    //         return {};
    //     }
    //     std::vector<BoostPolygon> mergedPolygons;
    //     for (const auto &poly : intersectionResult) {
    //         bool merged = false;
    //         for (auto &mergedPoly : mergedPolygons) {
    //             if (bg::intersects(poly, mergedPoly)) {
    //                 std::vector<BoostPolygon> unionResult;
    //                 bg::union_(mergedPoly, poly, unionResult);
    //                 if (!unionResult.empty()) {
    //                     mergedPoly = unionResult[0];
    //                     merged = true;
    //                     break;
    //                 }
    //             }
    //         }
    //         if (!merged) {
    //             mergedPolygons.push_back(poly);
    //         }
    //     }

    //     std::vector<std::shared_ptr<Polygon>> result;
    //     for (const auto &mergedPoly : mergedPolygons) {
    //         auto newPolygon = std::make_shared<Polygon>(mergedPoly);
    //         for (const auto &param : parameters) {
    //             newPolygon->setField(param.first, param.second);
    //         }
    //         result.push_back(newPolygon);
    //     }

    //     return result;
    // }

    std::string toWkt() const override { return convertToWkt(*polygon); }

    std::vector<std::pair<double, double>> getExterior() const {
        std::vector<std::pair<double, double>> result;
        for (const auto &point : bg::exterior_ring(*polygon)) {
            result.emplace_back(bg::get<0>(point), bg::get<1>(point));
        }
        return result;
    }

    std::vector<std::vector<std::pair<double, double>>> getInteriors() const {
        std::vector<std::vector<std::pair<double, double>>> result;
        for (const auto &inner : polygon->inners()) {
            std::vector<std::pair<double, double>> inner_result;
            for (const auto &point : inner) {
                inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
            }
            result.push_back(inner_result);
        }
        return result;
    }

    double getArea() const { return bg::area(*polygon); }

private:
    void setExterior(const std::vector<std::pair<double, double>> &coordinates) {
        bg::exterior_ring(*polygon).clear();
        for (const auto &coord : coordinates) {
            bg::append(*polygon, BoostPoint(coord.first, coord.second));
        }
        // Close the ring if it's not already closed
        if (coordinates.front() != coordinates.back()) {
            bg::append(*polygon, BoostPoint(coordinates.front().first, coordinates.front().second));
        }
    }

    void setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors) {
        bg::interior_rings(*polygon).clear();
        polygon->inners().resize(interiors.size());
        for (size_t i = 0; i < interiors.size(); ++i) {
            const auto &interior_coords = interiors[i];
            auto &inner = polygon->inners()[i];
            inner.clear();

            // Process the interior ring in reverse order
            for (auto it = interior_coords.rbegin(); it != interior_coords.rend(); ++it) {
                bg::append(inner, BoostPoint(it->first, it->second));
            }
            // Close the ring if it's not already closed
            if (interior_coords.front() != interior_coords.back()) {
                bg::append(inner, BoostPoint(interior_coords.back().first, interior_coords.back().second));
            }
        }
    }
};

class Point : public BaseGeometry {
public:
    ~Point() override = default;
    std::shared_ptr<BoostPoint> point;

    Point() : point(std::make_shared<BoostPoint>()) {}
    Point(const BoostPoint &p) : point(std::make_shared<BoostPoint>(p)) {}
    Point(std::shared_ptr<BoostPoint> p) : point(p) {}
    Point(double x, double y) : point(std::make_shared<BoostPoint>(x, y)) {}

    Point(const Point &other) : BaseGeometry(other), point(std::make_shared<BoostPoint>(*other.point)) {
        parameters = other.parameters; // Copy parameters
    }

    // Factory function for creating points from Python
    static std::shared_ptr<Point> create(double x, double y) { return std::make_shared<Point>(x, y); }

    std::string toWkt() const override { return convertToWkt(*point); }

    void setCoordinates(double x, double y) {
        bg::set<0>(*point, x);
        bg::set<1>(*point, y);
    }
    std::pair<double, double> getCoordinates() const { return std::make_pair(bg::get<0>(*point), bg::get<1>(*point)); }
    double getX() const { return bg::get<0>(*point); }
    double getY() const { return bg::get<1>(*point); }
    double distanceTo(const Point &other) const { return bg::distance(*point, *(other.point)); }
    bool equals(const Point &other) const { return bg::equals(*point, *(other.point)); }
    bool within(const Polygon &polygon) const { return bg::within(*point, *(polygon.polygon)); }

    std::shared_ptr<Point> centroid(const Polygon &polygon) const {
        BoostPoint centroid;
        bg::centroid(*(polygon.polygon), centroid);
        return std::make_shared<Point>(centroid);
    }

    double azimuth(const Point &other) const { return bg::azimuth(*point, *(other.point)); }

    std::shared_ptr<Point> translate(double dx, double dy) const {
        BoostPoint translated;
        bg::strategy::transform::translate_transformer<double, 2, 2> translate(dx, dy);
        bg::transform(*point, translated, translate);
        return std::make_shared<Point>(translated);
    }

    std::shared_ptr<Point> rotate(double angle, const Point &origin = Point(0, 0)) const {
        BoostPoint rotated;
        bg::strategy::transform::rotate_transformer<bg::degree, double, 2, 2> rotate(angle);
        bg::transform(*point, rotated, rotate);
        return std::make_shared<Point>(rotated);
    }

    std::shared_ptr<Point> scale(double scaling, const Point &origin = Point(0, 0)) const {
        BoostPoint scaled;
        double dx = getX() - origin.getX();
        double dy = getY() - origin.getY();

        bg::strategy::transform::scale_transformer<double, 2, 2> scale(scaling);
        bg::transform(BoostPoint(dx, dy), scaled, scale);

        return std::make_shared<Point>(scaled.get<0>() + origin.getX(), scaled.get<1>() + origin.getY());
    }
};

class GeometryContainer {
public:
    std::vector<std::shared_ptr<Polygon>> polygons;
    std::vector<std::shared_ptr<Point>> points;
    bgi::rtree<std::pair<BoostBox, size_t>, bgi::quadratic<16>> rtree;


    static void setPolygonFactory(py::function factory) {
        polygonFactory() = std::move(factory);
    }

    static void setPointFactory(py::function factory) {
        pointFactory() = std::move(factory);
    }

    // FactoryGuard creation functions for RAII management
    static FactoryGuard createPolygonFactoryGuard(py::function factory) {
        return FactoryGuard(polygonFactory(), factory);
    }

    static FactoryGuard createPointFactoryGuard(py::function factory) {
        return FactoryGuard(pointFactory(), factory);
    }

    void addPolygon(const std::shared_ptr<Polygon> &p) {
        // Print the parameters of the polygon being added
        BoostBox box;
        bg::envelope(*(p->polygon), box);
        rtree.insert(std::make_pair(box, polygons.size()));
        polygons.push_back(p);
    }

    void addPoint(const std::shared_ptr<Point> &p) {
        BoostBox box(*(p->point), *(p->point));
        rtree.insert(std::make_pair(box, polygons.size() + points.size()));
        points.push_back(p);
    }

    py::list getPolygons() {
        py::list py_polygons;
        for (const auto &polygon : polygons) {
            py_polygons.append(callFactoryFunction(polygon));
        }
        return py_polygons;
    }

    py::object readRegion(const std::pair<double, double> &coordinates, double scaling,
                          const std::pair<double, double> &size) {
        BoostPoint topLeft(coordinates.first, coordinates.second);
        BoostPoint bottomRight(coordinates.first + size.first / scaling, coordinates.second + size.second / scaling);
        BoostBox queryBox(topLeft, bottomRight);

        BoostPolygon intersectionPolygon;
        bg::convert(queryBox, intersectionPolygon);

        std::vector<std::pair<BoostBox, size_t>> results;
        rtree.query(bgi::intersects(queryBox), std::back_inserter(results));

        std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        py::list pyOutput;
        for (const auto &result : results) {
            size_t index = result.second;
            // Determine if the index corresponds to a polygon or a point
            // The insertation in the R-tree is done in the order of polygons and points so we can easily tell
            if (index < polygons.size()) {
                auto &polygon = polygons[index];
                auto intersectedPolygons = polygon->intersection(intersectionPolygon);
                for (const auto &intersectedPolygon : intersectedPolygons) {
                    GeometryUtils::applyAffineTransformation(*intersectedPolygon->polygon, coordinates, scaling);
                    pyOutput.append(callFactoryFunction(intersectedPolygon));
                }

            } else {
                auto &point = points[index - polygons.size()];
                // Let's make a copy before we apply the transformation, otherwise it will be changed in-place
                point = std::make_shared<Point>(*point);

                GeometryUtils::applyAffineTransformation(*point->point, coordinates, scaling);
                pyOutput.append(callFactoryFunction(point));
            }
        }

        return pyOutput;
    }

private:
    static py::function &polygonFactory() {
        static py::function instance;
        return instance;
    }

    static py::function &pointFactory() {
        static py::function instance;
        return instance;
    }

    // Call the appropriate factory function based on the type of the object
    py::object callFactoryFunction(const std::shared_ptr<Polygon> &polygon) {
        return invokeFactoryFunction(polygonFactory(), polygon);
    }

    py::object callFactoryFunction(const std::shared_ptr<Point> &point) {
        return invokeFactoryFunction(pointFactory(), point);
    }

    template <typename T>
    py::object invokeFactoryFunction(py::function factoryFunction, const std::shared_ptr<T> &object) {
        if (factoryFunction != py::function()) {
            try {
                py::object result = factoryFunction(object);
                if (result.ptr() != nullptr) {
                    return result;
                } else {
                    throw GeometryFactoryFunctionError("Factory function returned null object");
                }
            } catch (const std::exception &e) {
                throw GeometryFactoryFunctionError(std::string("Exception in factory function: ") + e.what());
            } catch (...) {
                throw GeometryFactoryFunctionError("Unknown exception in factory function");
            }
        }
        return py::cast(object);
    }
};

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
        .def("get_exterior", &Polygon::getExterior)
        .def("get_interiors", &Polygon::getInteriors)
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
        .def("get_x", &Point::getX)
        .def("get_y", &Point::getY)
        .def("distance_to", &Point::distanceTo)
        .def("equals", &Point::equals)
        .def("within", &Point::within)
        .def("centroid", &Point::centroid)
        .def("azimuth", &Point::azimuth)
        .def("translate", &Point::translate)
        .def("rotate", &Point::rotate, py::arg("angle"), py::arg("origin") = Point(0, 0))
        .def("scale", &Point::scale, py::arg("scaling"), py::arg("origin") = Point(0, 0))
        .def_property_readonly("wkt", &Point::toWkt);

    m.def("set_polygon_factory", &GeometryContainer::setPolygonFactory);
    m.def("set_point_factory", &GeometryContainer::setPointFactory);

    py::class_<GeometryContainer, std::shared_ptr<GeometryContainer>>(m, "GeometryContainer")
        .def(py::init<>())
        .def("add_polygon", &GeometryContainer::addPolygon)
        .def("add_point", &GeometryContainer::addPoint)
        .def("read_region", &GeometryContainer::readRegion)
        .def_property_readonly("polygons", &GeometryContainer::getPolygons)
        .def_property_readonly("points", [](const GeometryContainer &self) { return self.points; });

    py::register_exception<GeometryError>(m, "GeometryError");
    py::register_exception<GeometryIntersectionError>(m, "GeometryIntersectionError");
    py::register_exception<GeometryTransformationError>(m, "GeometryTransformationError");
    py::register_exception<GeometryFactoryFunctionError>(m, "GeometryFactoryFunctionError");
}