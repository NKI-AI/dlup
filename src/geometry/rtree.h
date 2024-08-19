#ifndef DLUP_GEOMETRY_RTREE_H
#define DLUP_GEOMETRY_RTREE_H
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <unordered_map>
#include <vector>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostBox = bg::model::box<BoostPoint>;

class RTreeBase {
  public:
  using RTreeType = bgi::rtree<std::pair<BoostBox, size_t>, bgi::quadratic<16>>;

  virtual ~RTreeBase() = default;

  virtual void rebuild() = 0; // Pure virtual function for rebuilding the R-tree

  void insert(const BoostBox &box, size_t index) {
    rtree.insert(std::make_pair(box, index));
    rTreeInvalidated = false;
  }

  template <typename QueryType, typename OutputIterator>
  void query(const QueryType &query, OutputIterator out) {
    if (rTreeInvalidated) {
      rebuild();
    }
    rtree.query(query, out);
  }

  void invalidate() { rTreeInvalidated = true; }

  void clear() {
    rtree.clear();
    rTreeInvalidated = true;
  }

  bool isInvalidated() const { return rTreeInvalidated; }

  protected:
  RTreeType rtree;
  bool rTreeInvalidated = true;
};

#endif // DLUP_GEOMETRY_RTREE_H
