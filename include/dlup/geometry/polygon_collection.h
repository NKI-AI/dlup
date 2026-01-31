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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_COLLECTION_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_COLLECTION_H_

#include <mutex>
#include <tuple>
#include <utility>
#include <vector>
#include "dlup/geometry/lazy_array.h"
#include "dlup/geometry/polygon.h"
#include "dlup/geometry/transforms.h"

namespace dlup::geometry {

class Polygon;

class PolygonCollection {
 public:
  PolygonCollection(std::vector<std::shared_ptr<Polygon>> polygons,
                    std::tuple<int, int> mask_size)
      : polygons_(std::move(polygons)),
        mask_size_(std::move(mask_size)),
        initialized_(true) {}

  // Lazy initialization constructor
  PolygonCollection(
      std::function<std::vector<std::shared_ptr<Polygon>>()> initializer,
      std::tuple<int, int> mask_size)
      : polygons_(),
        mask_size_(std::move(mask_size)),
        initialized_(false),
        initializer_(std::move(initializer)) {}

  std::vector<std::shared_ptr<Polygon>> GetGeometries() const {
    EnsureInitialized();
    return polygons_;
  }

  auto GetMaskSize() const { return mask_size_; }

  LazyArray<int> ToMask(int default_value = 0) const {
    EnsureInitialized();
    auto polygons_copy = polygons_;
    auto mask_size_copy = mask_size_;

    return LazyArray<int>(
        [polygons_copy, mask_size_copy, default_value]() -> std::vector<int> {
          return GenerateMaskFromAnnotations(polygons_copy, mask_size_copy,
                                             default_value);
        },
        std::vector<std::size_t>{
            static_cast<std::size_t>(std::get<1>(mask_size_copy)),
            static_cast<std::size_t>(std::get<0>(mask_size_copy))});
  }

  const std::vector<std::shared_ptr<Polygon>>& GetPolygons() const {
    EnsureInitialized();
    return polygons_;
  }

 private:
  void EnsureInitialized() const {
    if (!initialized_ && initializer_) {
      polygons_ =
          initializer_();  // Execute the lambda to initialize the polygons
      initialized_ = true;
    } else {
      // Do nothing if already initialized or no initializer provided
    }
  }

  mutable std::vector<std::shared_ptr<Polygon>> polygons_;
  std::tuple<int, int> mask_size_;
  mutable bool initialized_;
  std::function<std::vector<std::shared_ptr<Polygon>>()> initializer_;
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_POLYGON_COLLECTION_H_
