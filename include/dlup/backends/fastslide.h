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
#ifndef AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FASTSLIDE_H_
#define AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FASTSLIDE_H_

#include <vips/vips8>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dlup/backends/abstract.h"
#include "fastslide/fastslide.h"

namespace dlup::backends {

/// @brief Metadata structure for FastSlide backend
struct FastSlideMetadata : public SlideMetadata {
  std::string format_name;  ///< Format name (e.g., "QPTIFF", "SVS", "MRXS")

  FastSlideMetadata() = default;

  /// @brief Override Clone to support polymorphic copying
  std::unique_ptr<SlideMetadata> Clone() const override {
    return std::make_unique<FastSlideMetadata>(*this);
  }
};

/// @brief FastSlide backend implementation
///
/// This backend wraps the fastslide library to provide slide reading
/// capabilities. FastSlide handles format detection, level loading, and
/// tile management internally, making this backend simpler than VipsSlide.
class FastSlideSlide : public AbstractSlideBackend {
 public:
  /// @brief Constructor from filename
  /// @param filename Path to slide file
  /// @throws std::runtime_error if slide cannot be opened
  explicit FastSlideSlide(const fs::path& filename)
      : AbstractSlideBackend(filename) {
    auto fastslide_metadata = std::make_shared<FastSlideMetadata>();
    fastslide_metadata->filename = filename;
    metadata_ = fastslide_metadata;

    // Initialize VIPS
    if (VIPS_INIT("fastslide_backend")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);

    // Create reader using global registry (auto-detects format)
    auto reader_or =
        fastslide::runtime::GetGlobalRegistry().CreateReader(filename.string());
    if (!reader_or.ok()) {
      throw std::runtime_error(aifocore::fmt::format(
          "Failed to open slide '{}': {}", filename.string(),
          reader_or.status().message()));
    }

    reader_ = std::move(reader_or).value();

    // Load metadata from fastslide reader
    LoadMetadata();
  }

  /// @brief Constructor from metadata
  /// @param metadata FastSlideMetadata to initialize from
  explicit FastSlideSlide(const FastSlideMetadata& metadata)
      : AbstractSlideBackend(metadata) {
    auto fastslide_metadata = std::make_shared<FastSlideMetadata>(metadata);
    metadata_ = fastslide_metadata;

    if (VIPS_INIT("fastslide_backend")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);

    // Create reader
    auto reader_or = fastslide::runtime::GetGlobalRegistry().CreateReader(
        metadata.filename.string());
    if (!reader_or.ok()) {
      throw std::runtime_error(aifocore::fmt::format(
          "Failed to open slide '{}': {}", metadata.filename.string(),
          reader_or.status().message()));
    }

    reader_ = std::move(reader_or).value();
  }

  /// @brief Destructor
  ~FastSlideSlide() override { Close(); }

  /// @brief Get magnification from slide properties
  /// @return Magnification value or std::nullopt if not available
  [[nodiscard]] std::optional<double> GetMagnification() const override {
    if (!reader_) {
      return std::nullopt;
    }

    const auto& properties = reader_->GetProperties();
    if (properties.objective_magnification > 0.0) {
      return properties.objective_magnification;
    }

    // Try metadata as fallback
    auto metadata = reader_->GetMetadata();
    return metadata.GetDouble("magnification", 0.0) > 0.0
               ? std::optional<double>(metadata.GetDouble("magnification"))
               : std::nullopt;
  }

  /// @brief Get vendor/scanner information
  /// @return Vendor string or std::nullopt if not available
  [[nodiscard]] std::optional<std::string> GetVendor() const override {
    if (!reader_) {
      return std::nullopt;
    }

    auto metadata = reader_->GetMetadata();
    if (metadata.contains("scanner_model")) {
      return metadata.GetString("scanner_model");
    }
    return std::nullopt;
  }

  /// @brief Get list of property keys
  /// @return Vector of property key names
  [[nodiscard]] std::vector<std::string> GetProperties() const override {
    if (!reader_) {
      return {};
    }

    auto metadata = reader_->GetMetadata();
    std::vector<std::string> keys;
    keys.reserve(metadata.size());

    for (const auto& [key, value] : metadata) {
      keys.push_back(key);
    }

    return keys;
  }

  /// @brief Read a region from the slide
  /// @param coordinates Top-left coordinates in level-native coordinate system
  /// @param level Pyramid level to read from
  /// @param size Size of the region to read
  /// @return VImage containing the region or error status
  [[nodiscard]] aifocore::Result<vips::VImage> ReadRegion(
      const aifocore::Size<int, 2>& coordinates, int level,
      const aifocore::Size<int, 2>& size) const override {
    if (!reader_) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kFailedPrecondition,
                                  "Reader is closed");
    }

    // Create region specification with level-native coordinates
    fastslide::RegionSpec region;
    region.top_left = {static_cast<uint32_t>(coordinates[0]),
                       static_cast<uint32_t>(coordinates[1])};
    region.size = {static_cast<uint32_t>(size[0]),
                   static_cast<uint32_t>(size[1])};
    region.level = level;

    // Read region from fastslide
    auto image_or = reader_->ReadRegion(region);
    if (!image_or.ok()) {
      return AIFOCORE_MAKE_STATUS(
          aifocore::StatusCode::kInternal,
          aifocore::fmt::format("Failed to read region: {}",
                                image_or.status().message()));
    }

    // Convert fastslide::Image to vips::VImage
    return FastSlideImageToVips(image_or.value());
  }

  /// @brief Close the reader and release resources
  void Close() override { reader_.reset(); }

 private:
  std::unique_ptr<fastslide::SlideReader>
      reader_;  ///< FastSlide reader instance

  /// @brief Load metadata from fastslide reader
  void LoadMetadata() {
    auto fastslide_metadata =
        std::dynamic_pointer_cast<FastSlideMetadata>(metadata_);
    if (!fastslide_metadata) {
      throw std::runtime_error("Invalid metadata type for FastSlideSlide");
    }

    // Get format name
    fastslide_metadata->format_name = reader_->GetFormatName();

    // Get level count
    metadata_->level_count = reader_->GetLevelCount();
    metadata_->Resize();

    // Extract MPP from properties
    const auto& properties = reader_->GetProperties();
    double mpp_x = properties.mpp[0];
    double mpp_y = properties.mpp[1];

    // Validate MPP
    auto status = CheckIfMppIsValid(mpp_x, mpp_y);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }

    // Populate level information
    for (int level = 0; level < GetLevelCount(); ++level) {
      auto level_info_or = reader_->GetLevelInfo(level);
      if (!level_info_or.ok()) {
        throw std::runtime_error(
            aifocore::fmt::format("Failed to get level info for level {}: {}",
                                  level, level_info_or.status().message()));
      }

      const auto& level_info = level_info_or.value();

      // Store dimensions
      metadata_->level_dimensions[level] = {
          static_cast<int>(level_info.dimensions[0]),
          static_cast<int>(level_info.dimensions[1])};

      // Store downsample
      metadata_->level_downsamples[level] = level_info.downsample_factor;

      // Calculate spacing for this level
      metadata_->level_spacings[level] = {mpp_x * level_info.downsample_factor,
                                          mpp_y * level_info.downsample_factor};
    }

    // Set slide bounds
    const auto& bounds = properties.bounds;
    const auto [width, height] = GetDimensions();

    metadata_->slide_bounds = {
        {static_cast<int>(bounds.x), static_cast<int>(bounds.y)},
        {static_cast<int>(bounds.width), static_cast<int>(bounds.height)}};

    metadata_->slide_geometry = {
        {width, height},
        {static_cast<int>(bounds.x), static_cast<int>(bounds.y)},
        {static_cast<int>(bounds.width), static_cast<int>(bounds.height)}};
  }

  /// @brief Convert fastslide::Image to vips::VImage
  ///
  /// Optimized conversion with fast path for already-interleaved data.
  /// VIPS expects interleaved (contiguous) format, so we only convert if
  /// needed.
  ///
  /// @param image FastSlide image to convert
  /// @return VImage or error status
  [[nodiscard]] aifocore::Result<vips::VImage> FastSlideImageToVips(
      const fastslide::Image& image) const {
    if (image.Empty()) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Cannot convert empty image");
    }

    const uint32_t width = image.GetWidth();
    const uint32_t height = image.GetHeight();
    const uint32_t channels = image.GetChannels();
    const auto dtype = image.GetDataType();
    const auto planar_config = image.GetPlanarConfig();

    // Map fastslide data type to VIPS band format
    VipsBandFormat vips_format;
    switch (dtype) {
      case fastslide::DataType::kUInt8:
        vips_format = VIPS_FORMAT_UCHAR;
        break;
      case fastslide::DataType::kUInt16:
        vips_format = VIPS_FORMAT_USHORT;
        break;
      case fastslide::DataType::kInt16:
        vips_format = VIPS_FORMAT_SHORT;
        break;
      case fastslide::DataType::kUInt32:
        vips_format = VIPS_FORMAT_UINT;
        break;
      case fastslide::DataType::kInt32:
        vips_format = VIPS_FORMAT_INT;
        break;
      case fastslide::DataType::kFloat32:
        vips_format = VIPS_FORMAT_FLOAT;
        break;
      case fastslide::DataType::kFloat64:
        vips_format = VIPS_FORMAT_DOUBLE;
        break;
      default:
        return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                    "Unsupported data type");
    }

    try {
      // FAST PATH: Already interleaved (contiguous) - direct copy to VIPS
      // This avoids the unnecessary Clone() that ToInterleaved() would do
      if (planar_config == fastslide::PlanarConfig::kContiguous) {
        vips::VImage vips_image = vips::VImage::new_from_memory_copy(
            image.GetData(), image.SizeBytes(), width, height, channels,
            vips_format);
        return vips_image;
      }

      // SLOW PATH: Need planar→interleaved conversion
      // ToInterleaved() uses optimized SIMD conversion for RGB uint8
      auto converted_image = image.ToInterleaved();
      vips::VImage vips_image = vips::VImage::new_from_memory_copy(
          converted_image->GetData(), converted_image->SizeBytes(), width,
          height, channels, vips_format);

      return vips_image;
    } catch (const vips::VError& e) {
      return AIFOCORE_MAKE_STATUS(
          aifocore::StatusCode::kInternal,
          aifocore::fmt::format("VIPS error during image conversion: {}",
                                e.what()));
    }
  }
};

}  // namespace dlup::backends

#endif  // AIFO_DLUP_INCLUDE_DLUP_BACKENDS_FASTSLIDE_H_
