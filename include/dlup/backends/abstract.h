// Copyright 2024 Jonas Teuwen. All Rights Reserved.
// Copyright 2025 Joren Brunekreef. All Rights Reserved.
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
#ifndef AIFO_DLUP_INCLUDE_DLUP_BACKENDS_ABSTRACT_H_
#define AIFO_DLUP_INCLUDE_DLUP_BACKENDS_ABSTRACT_H_

#include <vips/vips8>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "aifocore/concepts/numeric.h"
#include "aifocore/status/result.h"
#include "aifocore/utilities/fmt.h"
#include "dlup/slide_geometry.h"

using aifocore::Size;

namespace dlup::backends {

namespace fs = std::filesystem;

inline constexpr double REL_TOL_MPP_ANISOTROPY = 0.015;

using Dimensions = Size<int, 2>;
using Spacing = Size<double, 2>;

struct SlideMetadata {
  fs::path filename;
  int level_count = 0;
  std::vector<Dimensions> level_dimensions;
  std::vector<Spacing> level_spacings;
  std::vector<double> level_downsamples;
  std::pair<Dimensions, Dimensions> slide_bounds;
  SlideGeometry slide_geometry;
  bool apply_color_profile = false;

  SlideMetadata() = default;

  virtual ~SlideMetadata() = default;

  virtual std::unique_ptr<SlideMetadata> Clone() const {
    return std::make_unique<SlideMetadata>(*this);
  }

  bool CheckIfLevelExists(int level) const {
    return level >= 0 && level < level_count;
  }

  void Resize() {
    level_dimensions.resize(level_count);
    level_spacings.resize(level_count);
    level_downsamples.resize(level_count);
  }
};

inline aifocore::Status CheckIfMppIsValid(double mpp_x, double mpp_y) {
  if (mpp_x <= 0.0 || mpp_y <= 0.0) {
    return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                "MPP values must be greater than zero.");
  }
  // Check if the relative difference is within the specified tolerance
  if (std::abs(mpp_x - mpp_y) >
      REL_TOL_MPP_ANISOTROPY * std::max(mpp_x, mpp_y)) {
    return AIFOCORE_MAKE_STATUS(
        aifocore::StatusCode::kInvalidArgument,
        "Cannot handle slides with anisotropic MPP values. "
        "Received mpp values: " +
            std::to_string(mpp_x) + " and " + std::to_string(mpp_y) + ".");
  }
  return aifocore::Status::OkStatus();
}

class AbstractSlideBackend {
 public:
  explicit AbstractSlideBackend(const fs::path& filename)
      : metadata_(
            std::make_shared<SlideMetadata>()) {  // Initialize shared_ptr first
    metadata_->filename = filename;
  }

  // Constructor from SlideMetadata
  explicit AbstractSlideBackend(const SlideMetadata& metadata)
      : metadata_(std::make_shared<SlideMetadata>(metadata)) {}

  virtual ~AbstractSlideBackend() = default;

  [[nodiscard]] std::shared_ptr<SlideMetadata> GetMetadata() const {
    return metadata_;
  }

  [[nodiscard]] int GetLevelCount() const noexcept {
    return metadata_->level_count;
  }

  [[nodiscard]] fs::path GetFilename() const noexcept {
    return metadata_->filename;
  }

  [[nodiscard]] Dimensions GetDimensions() const {
    return metadata_->level_dimensions.at(0);
  }

  /**
   * @brief Get the geometry of the slide.
   *
   * @return The SlideGeometry struct containing size, offset, and bounds.
   */
  [[nodiscard]] SlideGeometry GetGeometry() const {
    return metadata_->slide_geometry;
  }

  /**
   * @brief Get the scaled geometry of the slide.
   *
   * @param scaling The scaling factor to apply.
   * @return The scaled SlideGeometry.
   */
  [[nodiscard]] SlideGeometry GetScaledGeometry(double scaling) const {
    return GetGeometry().Scaled(scaling);
  }

  [[nodiscard]] Dimensions GetLevelDimensions(int level) const {
    bool exists = metadata_->CheckIfLevelExists(level);
    if (!exists) {
      throw std::out_of_range(
          aifocore::fmt::format("Invalid level {} for slide with {} levels",
                                level, metadata_->level_count));
    }
    return metadata_->level_dimensions.at(level);
  }

  [[nodiscard]] std::vector<Dimensions> GetLevelDimensions() const {
    return metadata_->level_dimensions;
  }

  [[nodiscard]] Spacing GetLevelSpacing(int level) const {
    bool exists = metadata_->CheckIfLevelExists(level);
    if (!exists) {
      throw std::out_of_range(
          aifocore::fmt::format("Invalid level {} for slide with {} levels",
                                level, metadata_->level_count));
    }
    return metadata_->level_spacings[level];
  }

  [[nodiscard]] std::vector<Spacing> GetLevelSpacings() const noexcept {
    return metadata_->level_spacings;
  }

  [[nodiscard]] Spacing GetSpacing() const noexcept {
    return metadata_->level_spacings[0];
  }

  [[nodiscard]] double GetScaling(double mpp) const {
    const auto spacing = GetLevelSpacing(0);
    const double mpp_x = spacing[0];
    const double mpp_y = spacing[1];
    return (mpp_x + mpp_y) / 2 / mpp;
  }

  [[nodiscard]] std::vector<double> GetLevelDownsamples() const noexcept {
    return metadata_->level_downsamples;
  }

  [[nodiscard]] aifocore::Result<double> GetLevelDownsample(
      int level) const noexcept {
    bool exists = metadata_->CheckIfLevelExists(level);
    if (!exists) {
      return AIFOCORE_MAKE_STATUS(
          aifocore::StatusCode::kInvalidArgument,
          aifocore::fmt::format("Invalid level {} for slide with {} levels",
                                level, metadata_->level_count));
    }
    return metadata_->level_downsamples[level];
  }

  [[nodiscard]] std::pair<Dimensions, Dimensions> GetSlideBounds() const {
    return metadata_->slide_bounds;
  }

  aifocore::Status SetSpacing(const Spacing& value) {
    const double mpp_x = value[0];
    const double mpp_y = value[1];

    AIFOCORE_RETURN_IF_ERROR(CheckIfMppIsValid(mpp_x, mpp_y));

    metadata_->level_spacings.clear();
    std::ranges::transform(
        metadata_->level_downsamples,
        std::back_inserter(metadata_->level_spacings),
        [mpp_x, mpp_y](double downsample) {
          return Spacing{mpp_x * downsample, mpp_y * downsample};
        });
    return aifocore::Status::OkStatus();
  }

  [[nodiscard]] virtual std::optional<double> GetMagnification() const = 0;
  [[nodiscard]] virtual std::optional<std::string> GetVendor() const = 0;
  [[nodiscard]] virtual std::vector<std::string> GetProperties() const = 0;
  [[nodiscard]] virtual aifocore::Result<vips::VImage> ReadRegion(
      const aifocore::Size<int, 2>& coordinates, int level,
      const aifocore::Size<int, 2>& size) const = 0;
  virtual void Close() = 0;

  [[nodiscard]] int GetBestLevelForDownsample(
      double downsample) const noexcept {
    if (downsample <= 1.0) {
      return 0;
    }

    return static_cast<int>(std::ranges::distance(
               metadata_->level_downsamples.begin(),
               std::ranges::upper_bound(metadata_->level_downsamples,
                                        downsample))) -
           1;
  }

  [[nodiscard]] std::shared_ptr<vips::VImage> GetThumbnail(
      const aifocore::Size<int, 2>& requested_size) const {
    const int target_width = requested_size[0];
    const int target_height = requested_size[1];
    const auto dimensions = GetDimensions();
    const int base_width = dimensions[0];
    const int base_height = dimensions[1];

    if (target_width <= 0 || target_height <= 0) {
      throw std::invalid_argument(aifocore::fmt::format(
          "Invalid thumbnail dimensions: {}x{}", target_width, target_height));
    }

    const double downsample =
        std::max(static_cast<double>(base_width) / target_width,
                 static_cast<double>(base_height) / target_height);

    // Select appropriate level
    const int level = GetBestLevelForDownsample(downsample);
    const auto level_dimensions = GetLevelDimensions(level);
    const int level_width = level_dimensions[0];
    const int level_height = level_dimensions[1];

    try {
      // Read entire region at once
      // VIPS is demand-driven so this is fine
      aifocore::Result<vips::VImage> thumbnail =
          ReadRegion({0, 0}, level, {level_width, level_height});
      if (!thumbnail.ok()) {
        throw std::runtime_error("Failed to read region");
      }
      bool exists = metadata_->CheckIfLevelExists(level);
      if (!exists) {
        throw std::out_of_range(
            aifocore::fmt::format("Invalid level {} for slide with {} levels",
                                  level, metadata_->level_count));
      }

      // Calculate scale factor to fit within bounds
      // TODO(jonasteuwen): this needs a proper handling with a return status.
      const double scale_factor = std::min(
          static_cast<double>(target_width) / thumbnail.value().width(),
          static_cast<double>(target_height) / thumbnail.value().height());

      // Resize with calculated scale
      // TODO(jonasteuwen): this needs a proper handling with a return status.
      vips::VImage resized = thumbnail.value().resize(
          scale_factor, vips::VImage::option()->set("kernel", "lanczos3"));

      // Handle alpha channel if present
      if (resized.has_alpha()) {
        resized =
            resized.flatten(vips::VImage::option()->set("background", 255));
      }
      return std::make_shared<vips::VImage>(resized);
    } catch (const vips::VError& e) {
      throw std::runtime_error(
          aifocore::fmt::format("VIPS error creating thumbnail: {}", e.what()));
    }
  }

 protected:
  std::shared_ptr<SlideMetadata> metadata_;
};

}  // namespace dlup::backends

#endif  // AIFO_DLUP_INCLUDE_DLUP_BACKENDS_ABSTRACT_H_
