#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/simplify.hpp>
#include <boost/geometry/geometries/geometries.hpp>
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

class GeometryError : public std::runtime_error {
public:
    explicit GeometryError(const std::string &message) : std::runtime_error(message) {}
};

class GeometryIntersectionError : public GeometryError {
public:
    explicit GeometryIntersectionError(const std::string &message) : GeometryError(message) {}
};

class GeometryTransformationError : public GeometryError {
public:
    explicit GeometryTransformationError(const std::string &message) : GeometryError(message) {}
};

class GeometryFactoryFunctionError : public GeometryError {
public:
    explicit GeometryFactoryFunctionError(const std::string &message) : GeometryError(message) {}
};

// Function to make a polygon valid
BoostPolygon makeValid(const BoostPolygon &polygon) {
    BoostPolygon validPolygon = polygon;

    // Check if the polygon is valid
    if (!bg::is_valid(validPolygon)) {
        // Correct the polygon (removing self-intersections and duplicate points)
        bg::correct(validPolygon);

        // If still not valid, simplify it
        if (!bg::is_valid(validPolygon)) {
            BoostPolygon simplifiedPolygon;
            bg::simplify(validPolygon, simplifiedPolygon, 0.01); // Adjust tolerance as needed
            validPolygon = simplifiedPolygon;
        }
    }

    return validPolygon;
}

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
};

class Polygon : public BaseGeometry {
public:
    std::shared_ptr<BoostPolygon> polygon;

    Polygon() : polygon(std::make_shared<BoostPolygon>()) {}
    Polygon(const BoostPolygon &p) : polygon(std::make_shared<BoostPolygon>(p)) {}
    Polygon(std::shared_ptr<BoostPolygon> p) : polygon(p) {}

    Polygon(const std::vector<std::pair<double, double>> &exterior,
            const std::vector<std::vector<std::pair<double, double>>> &interiors = {})
        : polygon(std::make_shared<BoostPolygon>()) {
        setExterior(exterior);
        setInteriors(interiors);
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
        BoostPolygon validPolygon = makeValid(*polygon);
        BoostPolygon validOtherPolygon = makeValid(otherPolygon);

        std::vector<BoostPolygon> intersectionResult;
        bg::intersection(validPolygon, validOtherPolygon, intersectionResult);

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

    std::string toWkt() const {
        std::stringstream ss;
        ss << bg::wkt(*polygon);
        return ss.str();
    }

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

    ~Polygon() override = default;

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
    std::shared_ptr<BoostPoint> point;

    Point() : point(std::make_shared<BoostPoint>()) {}
    Point(const BoostPoint &p) : point(std::make_shared<BoostPoint>(p)) {}
    Point(std::shared_ptr<BoostPoint> p) : point(p) {}
    Point(double x, double y) : point(std::make_shared<BoostPoint>(x, y)) {}

    std::string toWkt() const {
        std::stringstream ss;
        ss << bg::wkt(*point);
        return ss.str();
    }

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

    // Static method to access the singleton instance of the factory function
    static py::function &pythonPolygonFactory() {
        static py::function instance; // Singleton instance, initialized only once
        return instance;
    }

    // Method to set the factory function
    static void setPolygonFactory(py::function factory) { pythonPolygonFactory() = factory; }

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
            py_polygons.append(callPolygonFactory(polygon));
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
                    applyAffineTransformation(*intersectedPolygon->polygon, coordinates, scaling);
                    pyOutput.append(callPolygonFactory(intersectedPolygon));
                }

            } else {
                auto &point = points[index - polygons.size()];
                applyAffineTransformation(*point->point, coordinates, scaling);
                // TODO: Factor
                pyOutput.append(point);
            }
        }

        return pyOutput;
    }

private:
    void applyAffineTransformation(BoostPolygon &polygon, const std::pair<double, double> &origin, double scaling) {
        bg::strategy::transform::matrix_transformer<double, 2, 2> transform(scaling, 0, -origin.first * scaling, 0,
                                                                            scaling, -origin.second * scaling, 0, 0, 1);

        // TODO: This is a bit weird that we can't just immediately apply this to the polygon
        // Apply the transformation to each point of the exterior ring
        for (auto &point : bg::exterior_ring(polygon)) {
            bg::transform(point, point, transform);
        }

        // Apply the transformation to each point of each interior ring
        for (auto &ring : bg::interior_rings(polygon)) {
            for (auto &point : ring) {
                bg::transform(point, point, transform);
            }
        }
    }

    void applyAffineTransformation(BoostPoint &point, const std::pair<double, double> &origin, double scaling) {
        double x = (bg::get<0>(point) - origin.first) * scaling;
        double y = (bg::get<1>(point) - origin.second) * scaling;
        bg::set<0>(point, x);
        bg::set<1>(point, y);
    }

    py::object callPolygonFactory(const std::shared_ptr<Polygon> &polygon) {
        if (pythonPolygonFactory() != py::function()) {
            try {
                py::object result = pythonPolygonFactory()(polygon);
                // Ensure the result is a valid Python object
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
        // Fallback to direct casting if factory function is not set
        return py::cast(polygon);
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
            auto new_polygon = std::make_shared<Polygon>(*other.polygon);
            new_polygon->parameters = other.parameters; // Copy the parameters
            return new_polygon;
        }))
        .def("get_exterior", &Polygon::getExterior)
        .def("get_interiors", &Polygon::getInteriors)
        .def_property_readonly("wkt", &Polygon::toWkt)
        .def_property_readonly("area", &Polygon::getArea);

    py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
        .def(py::init<>())
        .def(py::init<const BoostPoint &>())
        .def(py::init<double, double>())
        .def("to_wkt", &Point::toWkt)
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
        .def("scale", &Point::scale, py::arg("scaling"), py::arg("origin") = Point(0, 0));

    m.def("set_polygon_factory", &GeometryContainer::setPolygonFactory);

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