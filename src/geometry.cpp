#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>

#include <memory>
#include <vector>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace py = pybind11;

typedef bg::model::d2::point_xy<double> BoostPoint;
typedef bg::model::polygon<BoostPoint> BoostPolygon;
typedef bg::model::box<BoostPoint> BoostBox;
typedef bg::model::ring<BoostPoint> BoostRing;
typedef bg::model::linestring<BoostPoint> BoostLineString;
typedef bg::model::multi_polygon<BoostPolygon> BoostMultiPolygon;

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void setField(const std::string &name, py::object value) { parameters[name] = value; }

    py::object getField(const std::string &name) const {
        auto it = parameters.find(name);
        if (it != parameters.end()) {
            return it->second;
        }
        return py::none();
    }

    std::vector<std::string> getFields() const {
        std::vector<std::string> fieldNames;
        for (const auto &param : parameters) {
            fieldNames.push_back(param.first);
        }
        return fieldNames;
    }
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

    // static std::shared_ptr<Polygon> fromWkt(const std::string& wkt) {
    //     auto p = std::make_shared<Polygon>();
    //     bg::read_wkt(wkt, *(p->polygon));
    //     return p;
    // }

    
    // TODO: We don't just need to intersect with a box, but with any geometry
    std::vector<std::shared_ptr<Polygon>> intersection(const BoostBox& box) const {
        std::vector<BoostPolygon> intersection_result;
        bg::intersection(*polygon, box, intersection_result);

        std::vector<std::shared_ptr<Polygon>> result;
        for (const auto& intersected_boost_polygon : intersection_result) {
            auto intersected_polygon = std::make_shared<Polygon>(intersected_boost_polygon);
            
            // Copy the parameters from this polygon to the new one
            for (const auto& param : parameters) {
                intersected_polygon->setField(param.first, param.second);
            }
            
            result.push_back(intersected_polygon);
        }

        return result;
    }

    std::string toWkt() const {
        std::stringstream ss;
        ss << bg::wkt(*polygon);
        return ss.str();
    }

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
};

class Point : public BaseGeometry {
public:
    std::shared_ptr<BoostPoint> point;

    Point() : point(std::make_shared<BoostPoint>()) {}
    Point(const BoostPoint &p) : point(std::make_shared<BoostPoint>(p)) {}
    Point(std::shared_ptr<BoostPoint> p) : point(p) {}
    Point(double x, double y) : point(std::make_shared<BoostPoint>(x, y)) {}

    // static std::shared_ptr<Point> fromWkt(const std::string& wkt) {
    //     auto p = std::make_shared<Point>();
    //     bg::read_wkt(wkt, *(p->point));
    //     return p;
    // }

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

    void add_polygon(const std::shared_ptr<Polygon> &p) {
        BoostBox box;
        bg::envelope(*(p->polygon), box);
        rtree.insert(std::make_pair(box, polygons.size()));
        polygons.push_back(p);
    }

    void add_point(const std::shared_ptr<Point> &p) {
        BoostBox box(*(p->point), *(p->point));
        rtree.insert(std::make_pair(box, polygons.size() + points.size()));
        points.push_back(p);
    }

    py::object read_region(const std::pair<double, double> &coordinates, double scaling,
                           const std::pair<double, double> &size) {
        BoostPoint top_left(coordinates.first, coordinates.second);
        BoostPoint bottom_right(coordinates.first + size.first / scaling, coordinates.second + size.second / scaling);
        BoostBox query_box(top_left, bottom_right);

        // Convert BoostBox to BoostPolygon
        BoostPolygon query_polygon;
        bg::convert(query_box, query_polygon);

        std::vector<std::pair<BoostBox, size_t>> results;
        rtree.query(bgi::intersects(query_box), std::back_inserter(results));

        std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        // std::stringstream ss;
        // std::cout << "Query box: " << bg::wkt(query_box) << std::endl;

        py::list py_output;
        for (const auto &result : results) {
            size_t index = result.second;

            // Determine if the index corresponds to a polygon or a point
            // The insertation in the R-tree is done in the order of polygons and points so we can easily tell

            if (index < polygons.size()) {
                // Add the polygon to the output list
                auto &polygon = polygons[index];

                // std::cout << "Original Polygon WKT: " << bg::wkt(*(polygon->polygon)) << std::endl;

                // Use the new intersect method
                auto intersected_polygons = polygon->intersection(query_box);

                for (const auto &intersected_polygon : intersected_polygons) {
                    // std::cout << "Intersected Polygon WKT: " << intersected_polygon->toWkt() << std::endl;
                    py_output.append(callPolygonFactory(intersected_polygon));
                }

            } else {
                py_output.append(points[index - polygons.size()]);
            }
        }

        return py_output;
    }

private:
    void transform_geometry(std::shared_ptr<BaseGeometry> geom, const BoostPoint &origin, double scaling) {
        if (auto polygon = std::dynamic_pointer_cast<Polygon>(geom)) {
            bg::strategy::transform::scale_transformer<double, 2, 2> scale(scaling, scaling);
            bg::strategy::transform::translate_transformer<double, 2, 2> translate(-origin.get<0>(), -origin.get<1>());
            BoostPolygon transformed;
            bg::transform(*(polygon->polygon), transformed, scale);
            bg::transform(transformed, *(polygon->polygon), translate);
        } else if (auto point = std::dynamic_pointer_cast<Point>(geom)) {
            double x = (point->getX() - origin.get<0>()) * scaling;
            double y = (point->getY() - origin.get<1>()) * scaling;
            point->setCoordinates(x, y);
        }
    }
    py::object callPolygonFactory(const std::shared_ptr<Polygon> &polygon) {
        if (pythonPolygonFactory() != py::function()) {
            // std::cout << "Using factory function" << std::endl;
            try {
                py::object result = pythonPolygonFactory()(polygon);
                // Ensure the result is a valid Python object
                if (result.ptr() != nullptr) {
                    return result;
                } else {
                    std::cerr << "Factory function returned null object" << std::endl;
                }
            } catch (const std::exception &e) {
                std::cerr << "Exception in factory function: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown exception in factory function" << std::endl;
            }
        }
        // Fallback to direct casting if factory function fails or is not set
        return py::cast(polygon);
    }
};

PYBIND11_MODULE(_geometry, m) {
    py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
        .def("set_field", &BaseGeometry::setField)
        .def("get_field", &BaseGeometry::getField);

    py::class_<Polygon, BaseGeometry, std::shared_ptr<Polygon>>(m, "Polygon")
        .def(py::init<>())
        .def(py::init<const BoostPolygon &>())
        .def(py::init<const std::vector<std::pair<double, double>> &,
                      const std::vector<std::vector<std::pair<double, double>>> &>())
        .def(py::init([](const Polygon &other) { return std::make_shared<Polygon>(other.polygon); }))
        // .def_static("from_wkt", &Polygon::fromWkt)
        .def("to_wkt", &Polygon::toWkt)
        .def("set_exterior", &Polygon::setExterior)
        .def("set_interiors", &Polygon::setInteriors)
        .def("get_exterior", &Polygon::getExterior)
        .def("get_interiors", &Polygon::getInteriors)
        .def("get_area", &Polygon::getArea);

    py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
        .def(py::init<>())
        .def(py::init<const BoostPoint &>())
        .def(py::init<double, double>())
        // .def_static("from_wkt", &Point::fromWkt)
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
        .def("add_polygon", &GeometryContainer::add_polygon)
        .def("add_point", &GeometryContainer::add_point)
        .def("read_region", &GeometryContainer::read_region)
        .def_property_readonly("polygons", [](const GeometryContainer &self) { return self.polygons; })
        .def_property_readonly("points", [](const GeometryContainer &self) { return self.points; });
}