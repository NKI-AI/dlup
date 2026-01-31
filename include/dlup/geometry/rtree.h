// Copyright 2024 Jonas Teuwen. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_RTREE_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_RTREE_H_

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace dlup::geometry {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostBox = bg::model::box<BoostPoint>;

class RTreeBase {
 public:
  using RTreeType = bgi::rtree<std::pair<BoostBox, size_t>, bgi::quadratic<16>>;

  virtual ~RTreeBase() = default;

  virtual void
  Rebuild() = 0;  // Pure virtual function for rebuilding the R-tree

  void Insert(const BoostBox& box, size_t index) {
    std::lock_guard<std::mutex> lock(
        rtree_mutex_);  // Lock the mutex for thread safety
    rtree_.insert(std::make_pair(box, index));
    rtree_invalidated_ = false;
  }

  template <typename QueryType, typename OutputIterator>
  void Query(const QueryType& query, OutputIterator out) const {
    std::lock_guard<std::mutex> lock(
        rtree_mutex_);  // Lock the mutex for thread safety
    if (rtree_invalidated_) {
      throw std::runtime_error(
          "R-tree is invalidated. Please rebuild before querying.");
    }
    rtree_.query(query, out);
  }

  void Invalidate() {
    std::lock_guard<std::mutex> lock(
        rtree_mutex_);  // Lock the mutex for thread safety
    rtree_invalidated_ = true;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(
        rtree_mutex_);  // Lock the mutex for thread safety
    rtree_.clear();
    rtree_invalidated_ = true;
  }

  bool IsInvalidated() const { return rtree_invalidated_; }

 protected:
  RTreeType rtree_;
  bool rtree_invalidated_ = true;
  mutable std::mutex rtree_mutex_;  // Mutex to protect R-tree operations
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_RTREE_H_
