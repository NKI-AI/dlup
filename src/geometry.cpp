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

typedef bg::model::point<double, 2, bg::cs::cartesian> BoostPoint;
typedef bg::model::polygon<BoostPoint> BoostPolygon;
typedef bg::model::ring<BoostPoint> BoostRing;

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void setField(const std::string& name, py::object value) {
        parameters[name] = value;
    }

    py::object getField(const std::string& name) const {
        auto it = parameters.find(name);
        if (it != parameters.end()) {
            return it->second;
        }
        return py::none();
    }

    std::vector<std::string> getFields() const {
        std::vector<std::string> field_names;
        for (const auto& param : parameters) {
            field_names.push_back(param.first);
        }
        return field_names;
    }
};

class Polygon : public BaseGeometry {
public:
    std::shared_ptr<BoostPolygon> polygon;

    Polygon() : polygon(std::make_shared<BoostPolygon>()) {}
    Polygon(const BoostPolygon& p) : polygon(std::make_shared<BoostPolygon>(p)) {}
    Polygon(std::shared_ptr<BoostPolygon> p) : polygon(p) {}

    Polygon(const std::vector<std::pair<double, double>>& exterior,
                   const std::vector<std::vector<std::pair<double, double>>>& interiors = {})
        : polygon(std::make_shared<BoostPolygon>()) 
    {
        set_exterior(exterior);
        set_interiors(interiors);
    }

    static std::shared_ptr<Polygon> fromWkt(const std::string& wkt) {
        auto p = std::make_shared<Polygon>();
        bg::read_wkt(wkt, *(p->polygon));
        return p;
    }

    std::string toWkt() const {
        std::stringstream ss;
        ss << bg::wkt(*polygon);
        return ss.str();
    }

    void set_exterior(const std::vector<std::pair<double, double>>& coordinates) {
        bg::exterior_ring(*polygon).clear();
        for (const auto& coord : coordinates) {
            bg::append(*polygon, BoostPoint(coord.first, coord.second));
        }
        // Close the ring if it's not already closed
        if (coordinates.front() != coordinates.back()) {
            bg::append(*polygon, BoostPoint(coordinates.front().first, coordinates.front().second));
        }
    }

    void set_interiors(const std::vector<std::vector<std::pair<double, double>>>& interiors) {
        bg::interior_rings(*polygon).clear();
        polygon->inners().resize(interiors.size());
        for (size_t i = 0; i < interiors.size(); ++i) {
            const auto& interior_coords = interiors[i];
            auto& inner = polygon->inners()[i];
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

    std::vector<std::pair<double, double>> get_exterior() const {
        std::vector<std::pair<double, double>> result;
        for (const auto& point : bg::exterior_ring(*polygon)) {
            result.emplace_back(bg::get<0>(point), bg::get<1>(point));
        }
        return result;
    }

    std::vector<std::vector<std::pair<double, double>>> get_interiors() const {
        std::vector<std::vector<std::pair<double, double>>> result;
        for (const auto& inner : polygon->inners()) {
            std::vector<std::pair<double, double>> inner_result;
            for (const auto& point : inner) {
                inner_result.emplace_back(bg::get<0>(point), bg::get<1>(point));
            }
            result.push_back(inner_result);
        }
        return result;
    }

    double get_area() const {
        return bg::area(*polygon);
    }
};

class Point : public BaseGeometry {
public:
    BoostPoint point;

    Point() : point() {}
    Point(const BoostPoint& p) : point(p) {}
    Point(double x, double y) : point(x, y) {}
};

class GeometryContainer {
public:
    std::vector<std::shared_ptr<Polygon>> polygons;
    std::vector<std::shared_ptr<Point>> points;

    void add_polygon(const std::shared_ptr<Polygon>& p) {
        polygons.push_back(p);
    }

    void add_point(const std::shared_ptr<Point>& p) {
        points.push_back(p);
    }

    py::object read_region(const BoostPoint& coordinates, double scaling, double size) {
        // To implement.
    }
};

PYBIND11_MODULE(_geometry, m) {
    py::class_<BaseGeometry, std::shared_ptr<BaseGeometry>>(m, "BaseGeometry")
        .def("set_parameter", &BaseGeometry::setField)
        .def("get_parameter", &BaseGeometry::getField);

    py::class_<Polygon, BaseGeometry, std::shared_ptr<Polygon>>(m, "BoostPolygon")
        .def(py::init<>())
        .def(py::init<const BoostPolygon&>())
        .def(py::init<const std::vector<std::pair<double, double>>&,
                      const std::vector<std::vector<std::pair<double, double>>>&>())
        .def(py::init([](const Polygon& other) {
            return std::make_shared<Polygon>(other.polygon);
        }))
        .def_static("from_wkt", &Polygon::fromWkt)
        .def("to_wkt", &Polygon::toWkt)
        .def("set_exterior", &Polygon::set_exterior)
        .def("set_interiors", &Polygon::set_interiors)
        .def("get_exterior", &Polygon::get_exterior)
        .def("get_interiors", &Polygon::get_interiors)
        .def("get_area", &Polygon::get_area);

    py::class_<Point, BaseGeometry, std::shared_ptr<Point>>(m, "Point")
        .def(py::init<>())
        .def(py::init<const BoostPoint&>());

    py::class_<GeometryContainer, std::shared_ptr<GeometryContainer>>(m, "GeometryContainer")
        .def(py::init<>())
        .def("add_polygon", &GeometryContainer::add_polygon)
        .def("add_point", &GeometryContainer::add_point)
        .def("read_region", &GeometryContainer::read_region)
        .def_property_readonly("polygons", [](const GeometryContainer& self) { return self.polygons; })
        .def_property_readonly("points", [](const GeometryContainer& self) { return self.points; });
}