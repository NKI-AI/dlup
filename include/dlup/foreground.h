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
#ifndef AIFO_DLUP_INCLUDE_DLUP_FOREGROUND_H_
#define AIFO_DLUP_INCLUDE_DLUP_FOREGROUND_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "aifocore/concepts/numeric.h"
#include "aifocore/tiling/grid.h"
#include "dlup/geometry/collection.h"
#include "dlup/geometry/polygon_collection.h"

namespace dlup {

namespace tiling = aifocore::tiling;

/**
 * @brief A struct that holds foreground information and the original grid
 */
template <aifocore::Numeric T>
struct ForegroundResult {
  using IndexType = size_t;
  using IndicesType = std::vector<IndexType>;

  std::shared_ptr<const tiling::Grid<T>> unfiltered_grid;
  IndicesType foreground_indices;

  /**
   * @brief Construct a new Foreground Result object
   *
   * @param grid The original unfiltered grid
   * @param indices Vector of indices into the original grid representing
   * foreground tiles
   */
  ForegroundResult(std::shared_ptr<const tiling::Grid<T>> grid,
                   IndicesType&& indices)
      : unfiltered_grid(grid), foreground_indices(std::move(indices)) {}

  /**
   * @brief Get the foreground indices
   * @return The vector of foreground indices
   */
  const IndicesType& GetForegroundIndices() const { return foreground_indices; }

  /**
   * @brief Convert to a new Grid containing only the foreground tiles
   */
  tiling::Grid<T> ToGrid() const {
    std::vector<typename tiling::Grid<T>::Coordinate> coordinates;
    coordinates.reserve(foreground_indices.size());
    for (const auto& idx : foreground_indices) {
      coordinates.push_back((*unfiltered_grid)[idx]);
    }
    return tiling::Grid<T>::FromCoordinates(coordinates,
                                            unfiltered_grid->GetGridOrder());
  }
};

/**
 * @class Foreground
 * @brief Provides methods for filtering a grid of tiles based on coverage
 * thresholds.
 */
template <aifocore::Numeric T>
class Foreground {
 public:
  /**
   * @brief Filters a grid of tiles based on the average coverage of regions in
   * a GeometryCollection.
   *
   * This method filters the grid by reading regions corresponding to each grid
   * tile in the `GeometryCollection`, calculating their average coverage, and
   * keeping only the tiles that meet the specified threshold.
   *
   * @param grid The grid of tiles to filter.
   * @param collection The GeometryCollection object used to compute regions.
   * @param tile_size The size of each tile in pixels (width, height).
   * @param scaling The scaling factor to use when interpreting the grid
   * coordinates.
   * @param threshold (Optional) The coverage threshold. Only tiles with total
   * region area greater than this value will be kept. If no value is provided
   *                  or the value is negative, all tiles are kept. Value is in
   * the range [0, 1].
   * @return A ForegroundResult object containing the unfiltered grid and
   * foreground indices.
   */
  static ForegroundResult<T> FilterGrid(
      const tiling::Grid<T>& grid,
      const geometry::GeometryCollection& collection,
      const aifocore::Size<T, 2>& tile_size, double scaling,
      std::optional<double> threshold = std::nullopt);
};

template <aifocore::Numeric T>
ForegroundResult<T> Foreground<T>::FilterGrid(
    const tiling::Grid<T>& grid, const geometry::GeometryCollection& collection,
    const aifocore::Size<T, 2>& tile_size, double scaling,
    std::optional<double> threshold) {

  typename ForegroundResult<T>::IndicesType filtered_indices;
  filtered_indices.reserve(grid.Length());

  bool has_rois = collection.NumRois() > 0;
  if (has_rois) {
  } else {
  }

  std::shared_ptr<geometry::PolygonCollection> polygon_collection;

  for (size_t idx = 0; idx < grid.Length(); ++idx) {
    const auto& coord = grid[idx];
    aifocore::Size<T, 2> top_left = {static_cast<T>(coord[0] / scaling),
                                     static_cast<T>(coord[1] / scaling)};
    aifocore::Size<T, 2> scaled_tile_size = tile_size / scaling;

    auto region =
        collection.ReadRegion({top_left[0], top_left[1]}, scaling,
                              {scaled_tile_size[0], scaled_tile_size[1]});

    // Initialize polygon collection with the correct region data
    if (has_rois) {
      polygon_collection = region.GetRois();
    } else {
      polygon_collection = region.GetPolygons();
    }
    // TODO(jonasteuwen): We need to handle potential overlaps.
    double total_area = 0.0;
    for (const auto& polygon : polygon_collection->GetPolygons()) {
      total_area += polygon->GetArea();
    }

    double average = total_area / static_cast<double>(scaled_tile_size[0] *
                                                      scaled_tile_size[1]);
    if (!threshold.has_value() || threshold.value() < 0 ||
        average > *threshold) {
      filtered_indices.push_back(idx);
    }
  }

  return ForegroundResult<T>(std::make_shared<tiling::Grid<T>>(grid),
                             std::move(filtered_indices));
}

}  // namespace dlup

#endif  // AIFO_DLUP_INCLUDE_DLUP_FOREGROUND_H_
