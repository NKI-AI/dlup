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
#ifndef AIFO_DLUP_INCLUDE_DLUP_BACKENDS_VIPS_H_
#define AIFO_DLUP_INCLUDE_DLUP_BACKENDS_VIPS_H_

#include <vips/vips8>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "aifocore/utilities/fmt.h"
#include "dlup/backends/abstract.h"

constexpr auto OPENSLIDE_DOWNSAMPLE = "openslide.level[{}].downsample";
constexpr auto OPENSLIDE_LEVEL_HEIGHT = "openslide.level[{}].height";
constexpr auto OPENSLIDE_LEVEL_WIDTH = "openslide.level[{}].width";
constexpr auto OPENSLIDE_MPP_X = "openslide.mpp-x";
constexpr auto OPENSLIDE_MPP_Y = "openslide.mpp-y";
constexpr auto OPENSLIDE_VENDOR = "openslide.vendor";
constexpr auto OPENSLIDE_LEVEL_COUNT = "openslide.level-count";
constexpr auto OPENSLIDE_COMMENT = "openslide.comment";
constexpr auto OPENSLIDE_BOUNDS_X = "openslide.bounds-x";
constexpr auto OPENSLIDE_BOUNDS_Y = "openslide.bounds-y";
constexpr auto OPENSLIDE_BOUNDS_WIDTH = "openslide.bounds-width";
constexpr auto OPENSLIDE_BOUNDS_HEIGHT = "openslide.bounds-height";

namespace dlup::backends {

struct VipsSlideMetadata : public SlideMetadata {
  bool load_with_openslide = false;
  bool rgb = false;
  std::string loader;
  bool apply_color_profile = false;

  VipsSlideMetadata() = default;

  // Override Clone to support polymorphic copying
  std::unique_ptr<SlideMetadata> Clone() const override {
    return std::make_unique<VipsSlideMetadata>(*this);
  }
};

class VipsSlide : public AbstractSlideBackend {
 public:
  explicit VipsSlide(const fs::path& filename, bool load_with_openslide = false,
                     bool rgb = false, bool apply_color_profile = false)
      : AbstractSlideBackend(filename) {
    auto vips_metadata = std::make_shared<VipsSlideMetadata>();
    vips_metadata->filename = filename;
    vips_metadata->load_with_openslide = load_with_openslide;
    vips_metadata->rgb = rgb;
    vips_metadata->apply_color_profile = apply_color_profile;
    metadata_ = vips_metadata;

    if (VIPS_INIT("vips_slide")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);

    vips::VImage image =
        load_with_openslide
            ? vips::VImage::openslideload(GetFilename().c_str(), GetOptions())
            : vips::VImage::new_from_file(GetFilename().c_str(), GetOptions());

    images_[0] = ApplyColorProfileIfNeeded(image);

    auto vips_metadata_ptr =
        std::dynamic_pointer_cast<VipsSlideMetadata>(metadata_);
    vips_metadata_ptr->loader = load_with_openslide
                                    ? "openslideload"
                                    : images_[0].get_string("vips-loader");

    LoadMetadata();
  }

  explicit VipsSlide(const VipsSlideMetadata& metadata)
      : AbstractSlideBackend(metadata) {
    auto vips_metadata = std::make_shared<VipsSlideMetadata>(metadata);
    metadata_ = vips_metadata;

    if (VIPS_INIT("vips_slide")) {
      vips_error_exit(nullptr);
    }
    // Force single-threaded mode (1 = single thread, 0 = auto-detect)
    vips_concurrency_set(1);
    vips_cache_set_max(0);
  }

  ~VipsSlide() override { Close(); }

  [[nodiscard]] std::optional<double> GetMagnification() const override {
    EnsureLevelLoaded(0);
    return GetField<double>("openslide.objective-power");
  }

  [[nodiscard]] std::optional<std::string> GetVendor() const override {
    EnsureLevelLoaded(0);
    return GetField<std::string>("openslide.vendor");
  }

  [[nodiscard]] std::vector<std::string> GetProperties() const override {
    std::vector<std::string> properties;
    if (HasField("openslide.comment")) {
      properties.push_back("openslide.comment");
    }
    return properties;
  }

  [[nodiscard]] aifocore::Result<vips::VImage> ReadRegion(
      const aifocore::Size<int, 2>& coordinates, int level,
      const aifocore::Size<int, 2>& size) const override {

    // 1. Ensure level is loaded
    EnsureLevelLoaded(level);
    const vips::VImage& level_image = images_.at(level);

    // 2. Bounds checking (coordinates are already level-native)
    if (coordinates[0] < 0 || coordinates[1] < 0 ||
        coordinates[0] + size[0] > level_image.width() ||
        coordinates[1] + size[1] > level_image.height()) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Requested region is out of bounds.");
    }

    // 3. VIPS crop operation
    vips::VImage region =
        level_image.crop(coordinates[0], coordinates[1], size[0], size[1]);
    return region;
  }

  void Close() override { images_.clear(); }

 private:
  mutable std::unordered_map<int, vips::VImage>
      images_;  // Cached images by level

  [[nodiscard]] bool HasField(const std::string& field) const noexcept {
    return images_.at(0).get_typeof(field.c_str()) != 0;
  }

  template <typename T>
  [[nodiscard]] std::optional<T> GetField(const std::string& field) const;

  void LoadMetadata() {
    auto vips_metadata =
        std::dynamic_pointer_cast<VipsSlideMetadata>(metadata_);
    if (!vips_metadata) {
      throw std::runtime_error("Invalid metadata type for VipsSlide");
    }

    if (vips_metadata->loader == "openslideload") {
      ReadOpenSlideMetadata();
    } else if (vips_metadata->loader == "tiffload") {
      ReadTiffMetadata();
    } else {
      throw std::runtime_error("Unsupported loader: " + vips_metadata->loader);
    }
  }

  void EnsureLevelLoaded(int level) const {
    auto vips_metadata =
        std::dynamic_pointer_cast<VipsSlideMetadata>(metadata_);
    if (!vips_metadata) {
      throw std::runtime_error("Invalid metadata type for VipsSlide");
    }

    if (images_.find(level) == images_.end()) {
      // Load specific level based on loader type
      if (vips_metadata->loader == "openslideload") {
        images_[level] = ApplyColorProfileIfNeeded(vips::VImage::openslideload(
            GetFilename().c_str(), GetOptions(level)));
      } else if (vips_metadata->loader == "tiffload") {
        images_[level] = ApplyColorProfileIfNeeded(
            vips::VImage::tiffload(GetFilename().c_str(), GetOptions(level)));
      } else {
        throw std::runtime_error("Unsupported loader: " +
                                 vips_metadata->loader);
      }
    }
  }

  void ReadOpenSlideMetadata() {
    EnsureLevelLoaded(0);
    metadata_->level_count = GetField<int>(OPENSLIDE_LEVEL_COUNT).value_or(1);
    metadata_->Resize();

    auto mpp_x = GetField<double>(OPENSLIDE_MPP_X).value_or(0.0);
    auto mpp_y = GetField<double>(OPENSLIDE_MPP_X).value_or(0.0);

    auto status = CheckIfMppIsValid(mpp_x, mpp_y);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }

    for (int level = 0; level < GetLevelCount(); ++level) {
      std::string level_height_key =
          aifocore::fmt::format(OPENSLIDE_LEVEL_HEIGHT, level);
      std::string level_width_key =
          aifocore::fmt::format(OPENSLIDE_LEVEL_WIDTH, level);
      std::string downsample_key =
          aifocore::fmt::format(OPENSLIDE_DOWNSAMPLE, level);

      if (!HasField(level_width_key) || !HasField(level_width_key)) {
        throw std::runtime_error(
            "Missing OPENSLIDE_LEVEL_HEIGHT or OPENSLIDE_LEVEL_WIDTH");
      }

      metadata_->level_dimensions[level] = {
          static_cast<int>(GetField<double>(level_width_key).value()),
          static_cast<int>(GetField<double>(level_height_key).value())};

      metadata_->level_downsamples[level] =
          GetField<double>(downsample_key).value_or(1 << level);
      metadata_->level_spacings[level] = {
          mpp_x * metadata_->level_downsamples[level],
          mpp_y * metadata_->level_downsamples[level]};
    }

    // Setting metadata for the slide bounds
    int bounds_x = GetField<int>(OPENSLIDE_BOUNDS_X).value_or(0);
    int bounds_y = GetField<int>(OPENSLIDE_BOUNDS_Y).value_or(0);
    const auto [width, height] = GetDimensions();
    int bounds_width = GetField<int>(OPENSLIDE_BOUNDS_WIDTH).value_or(width);
    int bounds_height = GetField<int>(OPENSLIDE_BOUNDS_HEIGHT).value_or(height);

    // Validate and correct bounds to ensure they fit within slide dimensions
    bool exceeds_width = (bounds_x + bounds_width) > width;
    bool exceeds_height = (bounds_y + bounds_height) > height;

    if (exceeds_width) {
      // Clip bounds_width to fit within remaining slide width
      bounds_width = std::min(bounds_width, width - bounds_x);
    }

    if (exceeds_height) {
      // Clip bounds_height to fit within remaining slide height
      bounds_height = std::min(bounds_height, height - bounds_y);
    }

    metadata_->slide_bounds = {{bounds_x, bounds_y},
                               {bounds_width, bounds_height}};
    metadata_->slide_geometry = {
        {width, height}, {bounds_x, bounds_y}, {bounds_width, bounds_height}};
  }

  void ReadTiffMetadata() {
    metadata_->level_count = GetField<int>("n-pages").value_or(1);
    metadata_->Resize();

    // libvips always returns the resolution in pixels per millimeter
    double x_res = GetField<double>("xres").value_or(0.0);
    double y_res = GetField<double>("yres").value_or(0.0);
    if (x_res <= 0.0 || y_res <= 0.0) {
      throw std::runtime_error("Invalid resolution values");
    }

    double mpp_x = 1000.0 / x_res;
    double mpp_y = 1000.0 / y_res;

    auto status = CheckIfMppIsValid(mpp_x, mpp_y);
    if (!status.ok()) {
      throw std::runtime_error(std::string(status.message()));
    }

    for (int level = 0; level < GetLevelCount(); ++level) {
      if (level == 0) {
        metadata_->level_dimensions[level] = {images_[0].width(),
                                              images_[0].height()};
      } else {
        auto level_image = vips::VImage::tiffload(
            GetFilename().c_str(), vips::VImage::option()->set("page", level));
        metadata_->level_dimensions[level] = {level_image.width(),
                                              level_image.height()};
      }

      // The downsample is the maximum of the base level / current level
      double downsample =
          std::max(static_cast<double>(metadata_->level_dimensions[0][0]) /
                       metadata_->level_dimensions[level][0],
                   static_cast<double>(metadata_->level_dimensions[0][1]) /
                       metadata_->level_dimensions[level][1]);

      metadata_->level_downsamples[level] = downsample;
      metadata_->level_spacings[level] = {
          mpp_x * metadata_->level_downsamples[level],
          mpp_y * metadata_->level_downsamples[level]};
    }
    // The slide bounds are the effective visible area of the slide
    // TIFF files do not have slide bounds, so we set it to the full slide
    metadata_->slide_bounds = {{0, 0}, GetDimensions()};
    metadata_->slide_geometry = {GetDimensions(), {0, 0}, GetDimensions()};
  }

  vips::VOption* GetOptions(int level = 0) const {
    vips::VOption* base_option = vips::VImage::option();

    if (!base_option) {
      throw std::runtime_error("Failed to create base option for VipsSlide");
    }
    auto vips_metadata =
        std::dynamic_pointer_cast<VipsSlideMetadata>(metadata_);

    // VipsForeignLoadTiffFile does not support optional argument
    // rgb if this is always set
    if (vips_metadata->rgb && !vips_metadata->load_with_openslide) {
      base_option->set("rgb", true);
    }
    if (!vips_metadata->load_with_openslide) {
      base_option->set("access", "sequential");
    }

    if (level > 0) {
      if (vips_metadata->loader == "openslideload") {
        base_option->set("level", level);
      } else if (vips_metadata->loader == "tiffload") {
        base_option->set("page", level);
      }
    }

    return base_option;  // Return the configured option
  }

  vips::VImage ApplyColorProfileIfNeeded(const vips::VImage& image) const {
    auto vips_metadata =
        std::dynamic_pointer_cast<VipsSlideMetadata>(metadata_);
    if (!vips_metadata) {
      throw std::runtime_error("Invalid metadata type for VipsSlide");
    }

    if (vips_metadata->apply_color_profile) {
      // Apply the ICC transformation to sRGB
      return image.icc_transform("srgb",
                                 vips::VImage::option()->set("embedded", true));
    }
    // Return the original image if no transformation is needed
    return image;
  }
};

template <typename T>
std::optional<T> VipsSlide::GetField(const std::string& field) const {
  EnsureLevelLoaded(0);
  auto image_ = &images_.at(0);
  try {
    if (!HasField(field)) {
      return std::nullopt;
    }

    if constexpr (std::is_same_v<T, std::string>) {
      return image_->get_string(field.c_str());
    } else if constexpr (std::is_same_v<T, double>) {
      return (image_->get_typeof(field.c_str()) == VIPS_TYPE_REF_STRING)
                 ? std::stod(image_->get_string(field.c_str()))
                 : image_->get_double(field.c_str());
    } else if constexpr (std::is_integral_v<T>) {
      return (image_->get_typeof(field.c_str()) == VIPS_TYPE_REF_STRING)
                 ? static_cast<T>(std::stoll(image_->get_string(field.c_str())))
                 : static_cast<T>(image_->get_int(field.c_str()));
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
      return image_->get_array_int(field.c_str());
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      return image_->get_array_double(field.c_str());
    }
  } catch (const vips::VError&) {
    // Optional: log the error here
  }
  return std::nullopt;
}

}  // namespace dlup::backends

#endif  // AIFO_DLUP_INCLUDE_DLUP_BACKENDS_VIPS_H_
