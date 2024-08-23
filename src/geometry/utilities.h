#ifndef DLUP_GEOMETRY_UTILITIES_H
#define DLUP_GEOMETRY_UTILITIES_H
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/correct.hpp>
#include <boost/geometry/algorithms/is_valid.hpp>
#include <boost/geometry/algorithms/simplify.hpp>
#include <boost/geometry/algorithms/transform.hpp>
#include <boost/geometry/geometries/geometries.hpp>

namespace utilities {

namespace bg = boost::geometry;

// Aliases for common types
using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostBox = bg::model::box<BoostPoint>;

// Function to make a polygon valid
BoostPolygon MakeValid(const BoostPolygon &polygon) {
  BoostPolygon valid_polygon = polygon;

  // Check if the polygon is valid
  if (!bg::is_valid(valid_polygon)) {
    // Correct the polygon (removing self-intersections and duplicate points)
    bg::correct(valid_polygon);

    // If still not valid, simplify it
    if (!bg::is_valid(valid_polygon)) {
      BoostPolygon simplifiedPolygon;
      // TODO: emit a warning
      bg::simplify(valid_polygon, simplifiedPolygon, 0.01); // TODO: Adjust tolerance
      valid_polygon = simplifiedPolygon;
    }
  }

  return valid_polygon;
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

void AffineTransform(BoostBox &box, const std::pair<double, double> &origin, double scaling) {
  bg::strategy::transform::matrix_transformer<double, 2, 2> transform(scaling, 0, -origin.first, 0, scaling,
                                                                      -origin.second, 0, 0, 1);

  // Apply the transformation to the min corner
  bg::transform(bg::return_envelope<BoostBox>(box), box, transform);
}
} // namespace utilities
// namespace geometry_utils

#endif // DLUP_GEOMETRY_UTILITIES_H
