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
#ifndef AIFO_DLUP_INCLUDE_DLUP_SLIDE_DATASET_H_
#define AIFO_DLUP_INCLUDE_DLUP_SLIDE_DATASET_H_

#include <vips/vips8>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "aifocore/status/result.h"
#include "aifocore/tiling/grid.h"
#include "dlup/foreground.h"
#include "dlup/slide_image.h"

namespace dlup {
namespace fs = std::filesystem;

/**
 * @brief Represents the dimensions as a point in 2D space with integer values.
 */

// TODO(jonasteuwen): Revisit the type for coordinates
struct DatasetSample {
  std::shared_ptr<vips::VImage> tile;  ///< The tile image data.
  aifocore::Size<int, 2> coordinates;  ///< Tile's top-left coordinates.
  std::string identifier;  ///< Identifier for the image containing the tile
};

/**
 * @brief SlideDataset class for handling a tiled representation of slide
 * images.
 *
 * Provides functionality to iterate over tiles in a slide image, retrieve
 * individual tiles, and handle padding or cropping to fit specific dimensions.
 */
class SlideDataset {
 public:
  /**
   * @brief Iterator class for iterating over the tiles in the dataset.
   */
  class Iterator {
   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = vips::VImage;
    using difference_type = std::ptrdiff_t;
    using pointer = vips::VImage*;
    using reference = vips::VImage&;

    /**
     * @brief Constructs an iterator.
     *
     * @param dataset Pointer to the parent dataset.
     * @param index Starting index of the iterator.
     */
    Iterator(SlideDataset* dataset, int index)
        : dataset_(dataset), current_index_(index) {}

    /**
     * @brief Advances the iterator (prefix increment).
     *
     * @return Reference to the updated iterator.
     */
    Iterator& operator++() {
      ++current_index_;
      return *this;
    }

    /**
     * @brief Advances the iterator (postfix increment).
     *
     * @return Copy of the iterator before incrementing.
     */
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++current_index_;
      return tmp;
    }

    /**
     * @brief Checks if two iterators are equal.
     *
     * @param other The iterator to compare with.
     * @return True if equal, otherwise false.
     */
    bool operator==(const Iterator& other) const {
      return current_index_ == other.current_index_;
    }

    /**
     * @brief Checks if two iterators are not equal.
     *
     * @param other The iterator to compare with.
     * @return True if not equal, otherwise false.
     */
    bool operator!=(const Iterator& other) const { return !(*this == other); }

    /**
     * @brief Dereferences the iterator to access the current tile.
     *
     * @return The StatusOr containing the DatasetSample if successful,
     * or an error status if the index is out of range or the tile cannot be
     * read.
     */
    aifocore::Result<DatasetSample> operator*() {
      return dataset_->GetTile(current_index_);
    }

   private:
    SlideDataset* dataset_;  ///< Pointer to the parent dataset.
    int current_index_;      ///< Current index in the dataset.
  };

  /**
   * @brief Constructs a Dataset object.
   *
   * @param slide The slide image object.
   * @param grid The grid defining tile layout.
   * @param mpp Microns per pixel scaling.
   * @param tile_size The dimensions of each tile.
   * @param crop Whether to crop tiles to exact dimensions.
   */
  // TODO(jonasteuwen): Consider if tile_size is actually
  // needed or if a Grid is fine?
  SlideDataset(std::shared_ptr<SlideImage> slide, const tiling::Grid<int>& grid,
               double mpp, const aifocore::Size<int, 2>& tile_size,
               bool crop = true)
      : slide_(slide),
        grid_(grid),
        mpp_(mpp),
        tile_size_(tile_size),
        crop_(crop) {
    geometry_ = slide_->GetGeometry().Scaled(GetScaling());
  }

  /**
   * @brief Returns an iterator to the beginning of the dataset.
   *
   * @return Iterator to the first tile.
   */
  Iterator begin() { return Iterator(this, 0); }

  /**
   * @brief Returns an iterator to the end of the dataset.
   *
   * @return Iterator past the last tile.
   */
  Iterator end() { return Iterator(this, Length()); }

  /**
   * @brief Provides a generator function to iterate over tiles.
   *
   * @return Lambda function to generate tiles by index.
   */
  auto GetGenerator() {
    return [this](int index) -> aifocore::Result<DatasetSample> {
      DatasetSample sample_result;
      AIFOCORE_ASSIGN_OR_RETURN(sample_result, this->GetTile(index));
      return sample_result;
    };
  }

  /**
   * @brief Sets the foreground filter to determine which tiles should be
   * processed
   *
   * When set, only tiles whose coordinates are in the foreground result will be
   * processed. The foreground result contains both the original grid and the
   * indices of tiles considered to be in the foreground.
   *
   * @param result The foreground result containing the original grid and
   * foreground indices
   */
  void SetForegroundFilter(const ForegroundResult<int>& result) {
    foreground_result_ = result;
    has_foreground_filter_ = true;
  }

  /**
   * @brief Clears any active foreground filtering
   *
   * After calling this method, all tiles in the grid will be processed,
   * regardless of their foreground status.
   */
  void ClearForegroundFilter() {
    has_foreground_filter_ = false;
    foreground_result_ = std::nullopt;
  }

  /**
   * @brief Returns the total number of tiles to be processed
   *
   * If foreground filtering is active, returns the number of tiles in the
   * foreground. Otherwise, returns the total number of tiles in the grid.
   *
   * @return Number of tiles (filtered by foreground if active)
   */
  [[nodiscard]] size_t Length() const {
    if (!has_foreground_filter_) {
      return grid_.Length();
    }
    return foreground_result_->foreground_indices.size();
  }

  /**
   * @brief Retrieves the tile at the specified index.
   *
   * @param index Index of the tile to retrieve.
   * @return A StatusOr containing the DatasetSample if successful,
   * or an error status if the index is out of range or the tile cannot be read.
   */
  aifocore::Result<DatasetSample> GetTile(int index) {
    if (index < 0 || static_cast<size_t>(index) >= Length()) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Index out of range.");
    }

    size_t grid_index = has_foreground_filter_
                            ? foreground_result_->foreground_indices[index]
                            : index;

    const auto coordinates = grid_[grid_index];

    const int x = coordinates[0];
    const int y = coordinates[1];

    // Calculate how much of the tile fits within the image bounds
    const int read_width =
        std::min(tile_size_[0], std::max(0, GetImageBounds()[0] - x));
    const int read_height =
        std::min(tile_size_[1], std::max(0, GetImageBounds()[1] - y));

    if (read_width <= 0 || read_height <= 0) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Tile dimensions would be zero or negative");
    }

    // Keep original coordinates but read only the valid portion
    // TODO(jonasteuwen): Figure out a way to be able to construct
    // the variable inside the AIFOCORE_ASSIGN_OR_RETURN
    std::shared_ptr<vips::VImage> tile;
    AIFOCORE_ASSIGN_OR_RETURN(
        tile,
        slide_->ReadRegion({static_cast<double>(x), static_cast<double>(y)},
                           GetScaling(), {read_width, read_height}));
    if (!tile) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Failed to read tile region");
    }

    // If we need to pad to full tile size
    if (!crop_ &&
        (read_width != tile_size_[0] || read_height != tile_size_[1])) {
      vips::VImage background =
          vips::VImage::black(tile_size_[0], tile_size_[1]);

      // Handle multi-band images
      for (int b = 1; b < tile->bands(); ++b) {
        background = background.bandjoin(
            vips::VImage::black(tile_size_[0], tile_size_[1])
                .cast(tile->format()));
      }

      // Insert at position (0,0)
      // the original tile content goes at the top-left,
      // and the rest is padded with the background
      tile = std::make_shared<vips::VImage>(background.insert(*tile, 0, 0));
    }

    return DatasetSample{
        .tile = tile,
        .coordinates = {x, y},
        .identifier = std::string(GetSlide()->GetIdentifier())};
  }

  /**
   * @brief Returns the grid associated with the dataset.
   *
   * @return Reference to the grid object.
   */
  const tiling::Grid<int>& GetGrid() const { return grid_; }

  /**
   * @brief Returns the microns per pixel (MPP) scaling.
   *
   * @return Microns per pixel.
   */
  double GetMpp() const { return mpp_; }

  /**
   * @brief Returns the tile size dimensions.
   *
   * @return Tile size as `Dimensions`.
   */
  const aifocore::Size<int, 2>& GetTileSize() const { return tile_size_; }

  /**
   * @brief Returns the scaling factor for the slide image.
   *
   * @return Scaling factor.
   */
  double GetScaling() const { return slide_->GetScaling(mpp_); }

  /**
   * @brief Returns the slide image associated with the dataset.
   *
   * @return Reference to the slide image.
   */
  std::shared_ptr<SlideImage> GetSlide() { return slide_; }

  /**
   * @brief Returns whether tiles are cropped to fit exact dimensions.
   *
   * @return True if cropping is enabled, otherwise false.
   */
  bool GetCrop() const { return crop_; }

  /**
   * @brief Returns the bounds of the slide image.
   *
   * @return Dimensions of the slide image bounds.
   */
  aifocore::Size<int, 2> GetImageBounds() const {
    return geometry_.offset + geometry_.bounds;
  }

 private:
  /**
   * @brief Calculates the actual tile size based on bounds.
   *
   * @param x X-coordinate of the tile.
   * @param y Y-coordinate of the tile.
   * @return Actual tile size dimensions.
   */
  aifocore::Size<int, 2> GetActualTileSize(int x, int y) const {
    return {std::min(tile_size_[0], GetImageBounds()[0] - x),
            std::min(tile_size_[1], GetImageBounds()[1] - y)};
  }

  std::shared_ptr<SlideImage>
      slide_;               ///< Slide image associated with the dataset.
  tiling::Grid<int> grid_;  ///< Grid defining tile layout.
  double mpp_;              ///< Microns per pixel scaling.
  aifocore::Size<int, 2> tile_size_;  ///< Dimensions of each tile.
  bool crop_;                     ///< Flag to determine if tiles are cropped.
  dlup::SlideGeometry geometry_;  ///< Geometry of the slide image.
  bool has_foreground_filter_{
      false};  ///< Whether foreground filtering is active
  std::optional<ForegroundResult<int>>
      foreground_result_;  ///< Optional foreground result for filtering
};

}  // namespace dlup

#endif  // AIFO_DLUP_INCLUDE_DLUP_SLIDE_DATASET_H_
