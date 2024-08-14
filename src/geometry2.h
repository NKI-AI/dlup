#ifndef GEOMETRY_H
#define GEOMETRY_H
#pragma once

#include <boost/geometry.hpp>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace bg = boost::geometry;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    void setField(const std::string &name, py::object value) { parameters[name] = value; }

    std::optional<py::object> getField(const std::string &name) const {
        if (auto it = parameters.find(name); it != parameters.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    auto getFields() const {
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
    // This doesn't work, but is probably
    // Polygon(BoostPolygon &&p) : polygon(std::make_shared<BoostPolygon>(std::move(p))) {}
    Polygon(std::shared_ptr<BoostPolygon> p) : polygon(p) {}

    Polygon(const std::vector<std::pair<double, double>> &exterior,
            const std::vector<std::vector<std::pair<double, double>>> &interiors = {})
        : polygon(std::make_shared<BoostPolygon>()) {
        setExterior(std::move(exterior));
        setInteriors(std::move(interiors));
    }

    // TODO: Box is probably sufficient.
    std::vector<std::shared_ptr<Polygon>> intersection(const BoostPolygon &otherPolygon) const;

    std::string toWkt() const override { return convertToWkt(*polygon); }

    std::vector<std::pair<double, double>> getExterior() const;
    std::vector<std::vector<std::pair<double, double>>> getInteriors() const;

    double getArea() const { return bg::area(*polygon); }

private:
    void setExterior(const std::vector<std::pair<double, double>> &coordinates);
    void setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors);
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
    inline double getX() const { return bg::get<0>(*point); }
    inline double getY() const { return bg::get<1>(*point); }
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

#endif // GEOMETRY_UTILITIES_H
