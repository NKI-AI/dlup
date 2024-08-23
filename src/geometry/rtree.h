#ifndef DLUP_GEOMETRY_RTREE_H
#define DLUP_GEOMETRY_RTREE_H
#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <mutex>
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
    std::lock_guard<std::mutex> lock(rtree_mutex_); // Lock the mutex for thread safety
    rtree_.insert(std::make_pair(box, index));
    rtree_invalidated_ = false;
  }

  template <typename QueryType, typename OutputIterator>
  void query(const QueryType &query, OutputIterator out) {
    std::lock_guard<std::mutex> lock(rtree_mutex_); // Lock the mutex for thread safety
    if (rtree_invalidated_) {
      rebuild();
    }
    rtree_.query(query, out);
  }

  void invalidate() {
    std::lock_guard<std::mutex> lock(rtree_mutex_); // Lock the mutex for thread safety
    rtree_invalidated_ = true;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(rtree_mutex_); // Lock the mutex for thread safety
    rtree_.clear();
    rtree_invalidated_ = true;
  }

  bool isInvalidated() const {
    std::lock_guard<std::mutex> lock(rtree_mutex_); // Lock the mutex for thread safety
    return rtree_invalidated_;
  }

  protected:
  RTreeType rtree_;
  bool rtree_invalidated_ = true;
  mutable std::mutex rtree_mutex_; // Mutex to protect R-tree operations
};

#endif // DLUP_GEOMETRY_RTREE_H
