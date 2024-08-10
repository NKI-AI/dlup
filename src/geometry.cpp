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

typedef bg::model::point<double, 2, bg::cs::cartesian> Point;
typedef bg::model::polygon<Point> BoostPolygon;
typedef bg::model::ring<Point> Ring;

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void set_parameter(const std::string& name, py::object value) {
        parameters[name] = value;
    }

    py::object get_parameter(const std::string& name) const {
        auto it = parameters.find(name);
        if (it != parameters.end()) {
            return it->second;
        }
        return py::none();
    }
};

class PolygonWrapper : public BaseGeometry {
public:
public:
    BoostPolygon polygon;
    PolygonWrapper() = default;
    PolygonWrapper(const BoostPolygon& p) : polygon(p) {}
    PolygonWrapper(const std::vector<std::pair<double, double>>& exterior,
                   const std::vector<std::vector<std::pair<double, double>>>& interiors = {}) {
        set_exterior(exterior);
        set_interiors(interiors);
    }

    void set_exterior(const std::vector<std::pair<double, double>>& coordinates) {
        bg::exterior_ring(polygon).clear();
        for (const auto& coord : coordinates) {
            bg::append(polygon, Point(coord.first, coord.second));
        }
        // Close the ring if it's not already closed
        if (coordinates.front() != coordinates.back()) {
            bg::append(polygon, Point(coordinates.front().first, coordinates.front().second));
        }
    }

    void set_interiors(const std::vector<std::vector<std::pair<double, double>>>& interiors) {
        polygon.inners().clear();
        for (const auto& interior_coords : interiors) {
            typename BoostPolygon::ring_type inner;
            for (const auto& coord : interior_coords) {
                bg::append(inner, Point(coord.first, coord.second));
            }
            // Close the ring if it's not already closed
            if (interior_coords.front() != interior_coords.back()) {
                bg::append(inner, Point(interior_coords.front().first, interior_coords.front().second));
            }
            polygon.inners().push_back(inner);
        }
    }

    std::vector<std::pair<double, double>> get_exterior() const {
        std::vector<std::pair<double, double>> result;
        for (const auto& point : bg::exterior_ring(polygon)) {
            result.emplace_back(bg::get<0>(point), bg::get<1>(point));
        }
        return result;
    }

    std::vector<std::vector<std::pair<double, double>>> get_interiors() const {
        std::vector<std::vector<std::pair<double, double>>> result;
        for (const auto& inner : polygon.inners()) {
            std::vector<std::pair<double, double>> inner_result;
            for (const auto& point : inner) {
                inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
            }
            result.push_back(inner_result);
        }
        return result;
    }

    double get_area() const {
        return bg::area(polygon);
    }
    std::string debug_print() const {
        std::stringstream ss;
        ss << "Exterior: ";
        for (const auto& point : bg::exterior_ring(polygon)) {
            ss << "(" << bg::get<0>(point) << "," << bg::get<1>(point) << ") ";
        }
        ss << std::endl;

        ss << "Number of inner rings: " << polygon.inners().size() << std::endl;

        for (size_t i = 0; i < polygon.inners().size(); ++i) {
            ss << "Interior " << i << ": ";
            for (const auto& point : polygon.inners()[i]) {
                ss << "(" << bg::get<0>(point) << "," << bg::get<1>(point) << ") ";
            }
            ss << std::endl;
        }

        double outer_area = bg::area(bg::exterior_ring(polygon));
        ss << "Outer ring area: " << outer_area << std::endl;

        double inner_area = 0;
        for (const auto& inner : polygon.inners()) {
            inner_area += bg::area(inner);
        }
        ss << "Total inner rings area: " << inner_area << std::endl;

        ss << "Calculated total area: " << outer_area - inner_area << std::endl;
        ss << "get_area() result: " << get_area() << std::endl;

        return ss.str();
    }

};

class PointWrapper : public BaseGeometry {
public:
    Point point;

    PointWrapper() : point() {}
    PointWrapper(const Point& p) : point(p) {}
    PointWrapper(double x, double y) : point(x, y) {}
};

class GeometryContainer {
public:
    std::vector<std::shared_ptr<PolygonWrapper>> polygons;
    std::vector<std::shared_ptr<PointWrapper>> points;

    void add_polygon(const std::shared_ptr<PolygonWrapper>& p) {
        polygons.push_back(p);
    }

    void add_point(const std::shared_ptr<PointWrapper>& p) {
        points.push_back(p);
    }

    py::object read_region(const Point& coordinates, double scaling, double size) {
        // Implementation remains the same as before
        // ...
    }
};

PYBIND11_MODULE(_geometry, m) {
    py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
        .def("set_parameter", &BaseGeometry::set_parameter)
        .def("get_parameter", &BaseGeometry::get_parameter);

    py::class_<PolygonWrapper, BaseGeometry, std::shared_ptr<PolygonWrapper>>(m, "BoostPolygon")
        .def(py::init<>())
        .def(py::init<const BoostPolygon&>())
        .def(py::init<const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<std::pair<double, double>>>&>())
        .def("get_area", &PolygonWrapper::get_area)
        .def("debug_print", &PolygonWrapper::debug_print)
        .def("set_exterior", &PolygonWrapper::set_exterior)
        .def("set_interiors", &PolygonWrapper::set_interiors)
        .def("get_exterior", &PolygonWrapper::get_exterior)
        .def("get_interiors", &PolygonWrapper::get_interiors);

    py::class_<PointWrapper, BaseGeometry, std::shared_ptr<PointWrapper>>(m, "Point")
        .def(py::init<>())
        .def(py::init<const Point&>());

    py::class_<GeometryContainer, std::shared_ptr<GeometryContainer>>(m, "GeometryContainer")
        .def(py::init<>())
        .def("add_polygon", &GeometryContainer::add_polygon)
        .def("add_point", &GeometryContainer::add_point)
        .def("read_region", &GeometryContainer::read_region)
        .def_property_readonly("polygons", [](const GeometryContainer& self) { return self.polygons; })
        .def_property_readonly("points", [](const GeometryContainer& self) { return self.points; });
}