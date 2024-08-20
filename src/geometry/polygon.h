
#ifndef DLUP_GEOMETRY_POLYGON_H
#define DLUP_GEOMETRY_POLYGON_H
#pragma once

#include "utilities.h"
#include <boost/geometry.hpp>
#include <string>
#include <vector>

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;

class Polygon : public BaseGeometry {
  public:
  using ExteriorRing = std::vector<BoostPoint> &;
  using InteriorRings = std::vector<BoostRing> &;

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

  bool equals(const Polygon &other) const {
    bool polygon_is_equal = bg::equals(*polygon, *(other.polygon));
    return parameters_ == other.parameters_ && polygon_is_equal;
  }

  // TODO: Box is probably sufficient.
  std::vector<std::shared_ptr<Polygon>> intersection(const BoostPolygon &otherPolygon) const;

  std::string toWkt() const override { return convertToWkt(*polygon); }

  std::vector<std::pair<double, double>> getExterior() const;
  std::vector<std::vector<std::pair<double, double>>> getInteriors() const;

  bool contains(const Polygon &other) const { return bg::within(*(other.polygon), *polygon); }
  bool isValid() const { return bg::is_valid(*polygon); }

  void makeValid() { *polygon = geometry_utils::MakeValid(*polygon); }

  ExteriorRing getExteriorAsIterator() { return bg::exterior_ring(*polygon); }
  InteriorRings getInteriorAsIterator() { return polygon->inners(); }

  double getArea() const {
    // Shapely reorients the polygon in memory if it is not oriented correctly, but keeps the coordinates
    // So we need to make a copy here to avoid modifying the original polygon
    if (!is_corrected_) {
      // Make a copy of the current polygon
      BoostPolygon new_polygon = *polygon;
      bg::correct(new_polygon); // Correct the copied polygon
      return bg::area(new_polygon);
    }

    return bg::area(*polygon);
  }

  void setExterior(const std::vector<std::pair<double, double>> &coordinates);
  void setInteriors(const std::vector<std::vector<std::pair<double, double>>> &interiors);
  void correctIfNeeded() const;
  void scale(double scaling);
  void simplifyPolygon(double tolerance);

  private:
  mutable bool is_corrected_ = false; // mutable allows modification in const methods
};

void Polygon::scale(double scaling) { geometry_utils::AffineTransform(*polygon, {0.0, 0.0}, scaling); }
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

  is_corrected_ = false; // Mark as not corrected. Correction reorients and closes
}
std::vector<std::shared_ptr<Polygon>> Polygon::intersection(const BoostPolygon &otherPolygon) const {
  // correctIfNeeded();
  // Make the polygon valid if needed before performing the intersection
  // TODO: This simplifies the polygon!!
  BoostPolygon validPolygon = geometry_utils::MakeValid(*polygon);

  std::vector<BoostPolygon> intersectionResult;
  bg::intersection(validPolygon, otherPolygon, intersectionResult);

  std::vector<std::shared_ptr<Polygon>> result;
  for (const auto &intersectedBoostPolygon : intersectionResult) {
    auto intersectedPolygon = std::make_shared<Polygon>(intersectedBoostPolygon);
    // Copy the parameters from this polygon to the new one

    for (const auto &param : parameters_) {
      intersectedPolygon->setField(param.first, param.second);
    }

    result.emplace_back(intersectedPolygon);
  }

  return result;
}
void Polygon::simplifyPolygon(double tolerance) { bg::simplify(*polygon, *polygon, tolerance); }
void Polygon::correctIfNeeded() const {
  if (!is_corrected_) {
    bg::correct(*polygon); // Dereference the shared pointer to apply the correction
    is_corrected_ = true;
  }
}

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

  is_corrected_ = false; // Mark as not corrected. Correction reorients and closes
}

#endif // DLUP_GEOMETRY_POLYGON_H
