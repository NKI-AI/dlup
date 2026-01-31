// Copyright 2025 Jonas Teuwen. All Rights Reserved.
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
#ifndef AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FIMAGE_H_
#define AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FIMAGE_H_

#include <vips/vips8>

#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dlup/backends/abstract.h"
#include "fim/sources/fastslide_source.h"

namespace dlup::backends {

/// @brief Metadata structure for FImage backend
struct FImageMetadata : public SlideMetadata {
  std::string format_name;  ///< Format name (e.g., "QPTIFF", "SVS", "MRXS")

  FImageMetadata() = default;

  /// @brief Override Clone to support polymorphic copying
  std::unique_ptr<SlideMetadata> Clone() const override {
    return std::make_unique<FImageMetadata>(*this);
  }
};

/// @brief FImage backend implementation
///
/// This backend wraps the fimage FastSlideSource to provide slide reading
/// capabilities. It uses the fimage API for tile-based access to whole slide
/// images and converts fim::Tile data to vips::VImage for compatibility with
/// the dlup backend interface.
class FImageSlide : public AbstractSlideBackend {
 public:
  /// @brief Constructor from filename
  /// @param filename Path to slide file
  /// @throws std::runtime_error if slide cannot be opened
  explicit FImageSlide(const fs::path& filename)
      : AbstractSlideBackend(filename) {
    auto fimage_metadata = std::make_shared<FImageMetadata>();
    fimage_metadata->filename = filename;
    metadata_ = fimage_metadata;

    // Initialize VIPS
    if (VIPS_INIT("fimage_backend")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);

    // Create fimage source
    try {
      source_ = std::make_unique<fim::FastSlideSource>(
          fim::FastSlideSource::Create(filename));
    } catch (const std::exception& e) {
      throw std::runtime_error(aifocore::fmt::format(
          "Failed to open slide '{}': {}", filename.string(), e.what()));
    }

    // Load metadata from fimage source
    LoadMetadata();
  }

  /// @brief Constructor from metadata
  /// @param metadata FImageMetadata to initialize from
  explicit FImageSlide(const FImageMetadata& metadata)
      : AbstractSlideBackend(metadata) {
    auto fimage_metadata = std::make_shared<FImageMetadata>(metadata);
    metadata_ = fimage_metadata;

    if (VIPS_INIT("fimage_backend")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);

    // Create fimage source
    try {
      source_ = std::make_unique<fim::FastSlideSource>(
          fim::FastSlideSource::Create(metadata.filename));
    } catch (const std::exception& e) {
      throw std::runtime_error(
          aifocore::fmt::format("Failed to open slide '{}': {}",
                                metadata.filename.string(), e.what()));
    }
  }

  /// @brief Destructor
  ~FImageSlide() override { Close(); }

  /// @brief Get magnification from slide properties
  /// @return Magnification value or std::nullopt if not available
  [[nodiscard]] std::optional<double> GetMagnification() const override {
    if (!source_) {
      return std::nullopt;
    }

    const auto* reader = source_->GetReader();
    if (!reader) {
      return std::nullopt;
    }

    const auto& properties = reader->GetProperties();
    if (properties.objective_magnification > 0.0) {
      return properties.objective_magnification;
    }

    // Try metadata as fallback
    auto metadata = reader->GetMetadata();
    return metadata.GetDouble("magnification", 0.0) > 0.0
               ? std::optional<double>(metadata.GetDouble("magnification"))
               : std::nullopt;
  }

  /// @brief Get vendor/scanner information
  /// @return Vendor string or std::nullopt if not available
  [[nodiscard]] std::optional<std::string> GetVendor() const override {
    if (!source_) {
      return std::nullopt;
    }

    const auto* reader = source_->GetReader();
    if (!reader) {
      return std::nullopt;
    }

    auto metadata = reader->GetMetadata();
    if (metadata.contains("scanner_model")) {
      return metadata.GetString("scanner_model");
    }
    return std::nullopt;
  }

  /// @brief Get list of property keys
  /// @return Vector of property key names
  [[nodiscard]] std::vector<std::string> GetProperties() const override {
    if (!source_) {
      return {};
    }

    const auto* reader = source_->GetReader();
    if (!reader) {
      return {};
    }

    auto metadata = reader->GetMetadata();
    std::vector<std::string> keys;
    keys.reserve(metadata.size());

    for (const auto& [key, value] : metadata) {
      keys.push_back(key);
    }

    return keys;
  }

  /// @brief Read a region from the slide as lazy fim::Tile
  /// @param coordinates Top-left coordinates in level-native coordinate system
  /// @param level Pyramid level to read from
  /// @param size Size of the region to read
  /// @return Lazy fim::Tile (data materializes on GetData())
  [[nodiscard]] aifocore::Result<fim::Tile> ReadRegionLazy(
      const aifocore::Size<int, 2>& coordinates, int level,
      const aifocore::Size<int, 2>& size) const {
    if (!source_) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kFailedPrecondition,
                                  "Source is closed");
    }

    // Use level-native coordinates directly
    int level_x = coordinates[0];
    int level_y = coordinates[1];

    // Create lazy tile with producer that reads from fimage
    // The actual reading happens only when GetData() is called on the tile
    auto producer = [this, level, level_x, level_y,
                     size]() -> std::vector<uint8_t> {
      auto level_view = source_->LevelView(level);
      auto tile = level_view.GetTile(level_x, level_y, size[0], size[1]);

      return std::move(tile.GetDataMut());
    };

    // Get dimensions info
    auto level_view = source_->LevelView(level);
    auto dims = level_view.GetDimensions();

    // Return lazy tile
    return fim::Tile(level_x, level_y, size[0], size[1], dims.channels,
                     std::move(producer), dims.layout);
  }

  /// @brief Read a region from the slide (vips interface for compatibility)
  /// @param coordinates Top-left coordinates in level-native coordinate system
  /// @param level Pyramid level to read from
  /// @param size Size of the region to read
  /// @return VImage containing the region
  [[nodiscard]] aifocore::Result<vips::VImage> ReadRegion(
      const aifocore::Size<int, 2>& coordinates, int level,
      const aifocore::Size<int, 2>& size) const override {
    // Get lazy tile
    auto tile_or = ReadRegionLazy(coordinates, level, size);
    if (!tile_or.ok()) {
      return tile_or.status();
    }

    fim::Tile tile = std::move(tile_or).value();

    // Materialize and convert to vips
    const auto& tile_data = tile.GetData();

    try {
      vips::VImage vips_image = vips::VImage::new_from_memory_copy(
          tile_data.data(), tile_data.size(), tile.width, tile.height,
          tile.channels, VIPS_FORMAT_UCHAR);
      return vips_image;
    } catch (const vips::VError& e) {
      return AIFOCORE_MAKE_STATUS(
          aifocore::StatusCode::kInternal,
          aifocore::fmt::format("VIPS error during tile conversion: {}",
                                e.what()));
    }
  }

  /// @brief Close the source and release resources
  void Close() override { source_.reset(); }

 private:
  std::unique_ptr<fim::FastSlideSource> source_;  ///< FImage source instance

  /// @brief Load metadata from fimage source
  void LoadMetadata() {
    auto fimage_metadata = std::dynamic_pointer_cast<FImageMetadata>(metadata_);
    if (!fimage_metadata) {
      throw std::runtime_error("Invalid metadata type for FImageSlide");
    }

    // Get format name
    fimage_metadata->format_name = source_->GetFormatName();

    // Get level count
    metadata_->level_count = source_->GetLevelCount();
    metadata_->Resize();

    // Extract MPP
    auto mpp = source_->GetMpp();
    double mpp_x = mpp[0];
    double mpp_y = mpp[1];

    // Validate MPP
    auto status = CheckIfMppIsValid(mpp_x, mpp_y);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }

    // Populate level information
    for (int level = 0; level < GetLevelCount(); ++level) {
      auto level_dims = source_->GetLevelDimensions(level);
      double downsample = source_->GetLevelDownsample(level);

      // Store dimensions
      metadata_->level_dimensions[level] = {level_dims.width,
                                            level_dims.height};

      // Store downsample
      metadata_->level_downsamples[level] = downsample;

      // Calculate spacing for this level
      metadata_->level_spacings[level] = {mpp_x * downsample,
                                          mpp_y * downsample};
    }

    // Set slide bounds from reader properties
    const auto* reader = source_->GetReader();
    if (reader) {
      const auto& bounds = reader->GetProperties().bounds;
      const auto [width, height] = GetDimensions();

      metadata_->slide_bounds = {
          {static_cast<int>(bounds.x), static_cast<int>(bounds.y)},
          {static_cast<int>(bounds.width), static_cast<int>(bounds.height)}};

      metadata_->slide_geometry = {
          {width, height},
          {static_cast<int>(bounds.x), static_cast<int>(bounds.y)},
          {static_cast<int>(bounds.width), static_cast<int>(bounds.height)}};
    } else {
      // Fallback: use full slide dimensions
      const auto [width, height] = GetDimensions();
      metadata_->slide_bounds = {{0, 0}, {width, height}};
      metadata_->slide_geometry = {{width, height}, {0, 0}, {width, height}};
    }
  }
};

}  // namespace dlup::backends

#endif  // AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FIMAGE_H_
