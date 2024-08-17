#ifndef GEOMETRY_H
#define GEOMETRY_H
#pragma once

#include <boost/geometry.hpp>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "geometry_utils.h"

namespace bg = boost::geometry;
namespace py = pybind11;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;

class BaseGeometry {
public:
    virtual ~BaseGeometry() = default;
    std::unordered_map<std::string, py::object> parameters;

    virtual void setField(const std::string &name, py::object value) { parameters[name] = value; }

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
    using ExteriorRing = std::vector<BoostPoint>&;
    using InteriorRings = std::vector<BoostRing>&;

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

    std::string toWkt() const override {
         return convertToWkt(*polygon); }

    std::vector<std::pair<double, double>> getExterior() const;
    std::vector<std::vector<std::pair<double, double>>> getInteriors() const;

    ExteriorRing getExteriorAsIterator() {
        return bg::exterior_ring(*polygon);
    }

    InteriorRings getInteriorAsIterator() {
        return polygon->inners();
    }


    double getArea() const { 
        // Shapely reorients the polygon in memory if it is not oriented correctly, but keeps the coordinates
        // So we need to make a copy here to avoid modifying the original polygon
        if (!isCorrected) {
            // Make a copy of the current polygon
            BoostPolygon newPolygon = *polygon;
            bg::correct(newPolygon);  // Correct the copied polygon
            return bg::area(newPolygon);
        }

        return bg::area(*polygon); 
    }

    void setExterior(const std::vector<std::pair<double, double>> &coordinates);
    void setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors);
    void correctIfNeeded() const;
    void scale(double scaling);
    void simplifyPolygon(double tolerance);
private:
    mutable bool isCorrected = false;  // mutable allows modification in const methods
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

    void scale(double scaling) {
        setCoordinates(getX() * scaling, getY() * scaling);
    }
};

#endif // GEOMETRY_H
