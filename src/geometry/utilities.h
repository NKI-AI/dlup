#ifndef DLUP_GEOMETRY_UTILITIES_H
#define DLUP_GEOMETRY_UTILITIES_H
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/simplify.hpp>
#include <boost/geometry/algorithms/transform.hpp>
#include <boost/geometry/geometries/geometries.hpp>

namespace GeometryUtils {

namespace bg = boost::geometry;

// Aliases for common types
using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;

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
      // TODO: emit a warning
      bg::simplify(validPolygon, simplifiedPolygon, 0.01); // TODO: Adjust tolerance
      validPolygon = simplifiedPolygon;
    }
  }

  return validPolygon;
}

void AffineTransform(BoostPolygon &polygon, const std::pair<double, double> &origin, double scaling) {
  bg::strategy::transform::matrix_transformer<double, 2, 2> transform(scaling, 0, -origin.first, 0, scaling,
                                                                      -origin.second, 0, 0, 1);

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

// Function to apply an affine transformation to a point
void AffineTransform(BoostPoint &point, const std::pair<double, double> &origin, double scaling) {
  double x = (bg::get<0>(point) - origin.first) * scaling;
  double y = (bg::get<1>(point) - origin.second) * scaling;
  bg::set<0>(point, x);
  bg::set<1>(point, y);
}

} // namespace GeometryUtils

#endif // DLUP_GEOMETRY_UTILITIES_H
