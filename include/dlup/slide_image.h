// Copyright 2025 Jonas Teuwen & Joren Brunekreef. All Rights Reserved.
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
#ifndef AIFO_DLUP_INCLUDE_DLUP_SLIDE_IMAGE_H_
#define AIFO_DLUP_INCLUDE_DLUP_SLIDE_IMAGE_H_

#include <vips/vips8>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "aifocore/concepts/numeric.h"
#include "dlup/backends/abstract.h"
#include "dlup/backends/vips.h"
#include "dlup/slide_geometry.h"

using aifocore::Size;

namespace dlup {

// Use enum class for type safety
enum class Resampling { kNearest, kLanczos };

// Use constexpr for compile-time evaluation
[[nodiscard]] constexpr auto ResamplingToVipsKernel(
    Resampling resampling) noexcept {
  return resampling == Resampling::kNearest ? VIPS_KERNEL_NEAREST
                                            : VIPS_KERNEL_LANCZOS3;
}

inline aifocore::Status CheckSizeAndLocation(const Size<double, 2>& size,
                                             const Size<double, 2>& location,
                                             const Size<int, 2>& level_size) {
  if (size[0] <= 0 || size[1] <= 0) {
    return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                "Size values must be greater than zero.");
  }
  if (location[0] < 0 || location[1] < 0 ||
      location[0] + size[0] > level_size[0] ||
      location[1] + size[1] > level_size[1]) {
    return AIFOCORE_MAKE_STATUS(
        aifocore::StatusCode::kInvalidArgument,
        "Requested region is outside level boundaries.");
  }
  return aifocore::Status::OkStatus();
}

class SlideImage {
 public:
  static aifocore::Result<std::unique_ptr<SlideImage>> Create(
      std::shared_ptr<backends::AbstractSlideBackend> wsi,
      const std::optional<std::string_view>& identifier = std::nullopt,
      Resampling interpolator = Resampling::kLanczos,
      const std::optional<Size<double, 2>>& overwrite_mpp = std::nullopt) {
    if (!wsi) {
      return AIFOCORE_MAKE_STATUS(aifocore::StatusCode::kInvalidArgument,
                                  "Backend is null.");
    }

    if (overwrite_mpp) {
      AIFOCORE_RETURN_IF_ERROR(wsi->SetSpacing(*overwrite_mpp),
                               "Cannot overwrite mpp");
    }

    const auto [mpp_x, mpp_y] = wsi->GetSpacing();
    if (mpp_x == 0.0 || mpp_y == 0.0) {
      return AIFOCORE_MAKE_STATUS(
          aifocore::StatusCode::kFailedPrecondition,
          "The spacing cannot be derived from the image and is "
          "not explicitly set.");
    }

    auto image = std::unique_ptr<SlideImage>(new SlideImage(
        std::move(wsi), identifier.has_value() ? std::string(*identifier) : "",
        interpolator, (mpp_x + mpp_y) / 2.0));

    return image;
  }

  // Use default for special member functions where appropriate
  ~SlideImage() = default;
  SlideImage(const SlideImage&) = delete;
  SlideImage& operator=(const SlideImage&) = delete;
  SlideImage(SlideImage&&) noexcept = default;
  SlideImage& operator=(SlideImage&&) noexcept = default;

  /**
   * @brief Closes the underlying backend resources for the slide image.
   *
   * This function releases resources associated with the backend, such as file
   * handles.
   */
  void Close() noexcept { wsi_->Close(); }

  /**
   * @brief Reads a region at a specific scaling level from the whole-slide
   * image.
   *
   * This function simplifies the extraction of regions from pyramidal images,
   * providing continuous interpolation for intermediate resolutions. It selects
   * the closest high-resolution level to extract a target region via
   * interpolation, optionally using the kLanczos or kNearest resampling method.
   *
   * The process includes:
   * - Mapping the desired region to a lower level.
   * - Adding extra pixels for interpolation support.
   * - Converting coordinates to the native level's coordinate system.
   * - Cropping the native resolution region to the target region.
   * - Rescaling the cropped region to the desired size.
   *
   * @param location The top-left corner (x, y) of the region in pixel
   * coordinates at the requested scaling level.
   * @param scaling The scaling factor relative to level 0.
   * @param size The size of the region to extract, in pixels.
   * @return A shared pointer to a VImage containing the extracted region.
   */
  [[nodiscard]] aifocore::Result<std::shared_ptr<vips::VImage>> ReadRegion(
      const Size<double, 2>& location, double scaling,
      const Size<int, 2>& size) const {
    const Size<double, 2> size_arr = static_cast<Size<double, 2>>(size);

    const auto level_size = GetScaledSize(scaling);
    AIFOCORE_RETURN_IF_ERROR(
        CheckSizeAndLocation(size_arr, location, level_size),
        "Invalid size or location");

    const int native_level = wsi_->GetBestLevelForDownsample(1.0 / scaling);
    // TODO(jonasteuwen): this should return a Size<double, 2>
    const auto native_level_size = wsi_->GetLevelDimensions(native_level);

    double native_level_downsample = 0.0;
    AIFOCORE_ASSIGN_OR_RETURN(native_level_downsample,
                              wsi_->GetLevelDownsample(native_level),
                              "Failed to get level downsample");

    const double native_scaling = scaling * native_level_downsample;

    const Size<double, 2> native_location = location / native_scaling;
    const Size<double, 2> native_size = size_arr / native_scaling;

    // Add extra pixels for interpolation (e.g., Lanczos)
    const double native_extra_pixels =
        (native_scaling > 1.0) ? 3.0 : std::ceil(3.0 / native_scaling);

    Size<double, 2> native_location_adapted =
        aifocore::Floor(native_location - native_extra_pixels);

    // TODO(jonasteuwen): clamping should be possible on floats too
    native_location_adapted =
        aifocore::Clamp(native_location_adapted, {0., 0.},
                        static_cast<Size<double, 2>>(native_level_size));

    Size<double, 2> level_zero_location_adapted =
        aifocore::Floor(native_location_adapted * native_level_downsample);

    native_location_adapted =
        level_zero_location_adapted / native_level_downsample;

    // Calculate the size before any clamping
    Size<double, 2> native_size_adapted =
        aifocore::Ceil(native_location + native_size + native_extra_pixels);

    // Clamp size to level boundaries
    native_size_adapted = aifocore::Min(
        native_size_adapted, static_cast<Size<double, 2>>(native_level_size) -
                                 native_location_adapted);

    // Ensure integer precision
    native_size_adapted = aifocore::Ceil(native_size_adapted);
    vips::VImage vips_region;

    // Convert to level-native coordinates for backend
    Size<int, 2> native_coordinates = {
        static_cast<int>(native_location_adapted[0]),
        static_cast<int>(native_location_adapted[1])};

    AIFOCORE_ASSIGN_OR_RETURN(
        vips_region,
        wsi_->ReadRegion(native_coordinates, native_level,
                         {static_cast<int>(native_size_adapted[0]),
                          static_cast<int>(native_size_adapted[1])}),
        "Failed to read region");

    // Fractional coordinates for cropping
    const Size<double, 2> fractional_coordinates =
        native_location - native_location_adapted;

    // Define the crop box
    const auto [crop_left, crop_top] =
        static_cast<Size<int, 2>>(aifocore::Floor(fractional_coordinates));

    const auto [crop_right, crop_bottom] = static_cast<Size<int, 2>>(
        aifocore::Floor(fractional_coordinates + native_size));

    // Crop the region
    auto cropped_region = vips_region.crop(
        crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top);

    // Resize the cropped region to the target size
    const double xscale = static_cast<double>(size[0]) / cropped_region.width();
    const double yscale =
        static_cast<double>(size[1]) / cropped_region.height();

    auto resized_region = cropped_region.resize(
        xscale, vips::VImage::option()
                    ->set("vscale", yscale)
                    ->set("kernel", ResamplingToVipsKernel(interpolator_)));

    return std::make_shared<vips::VImage>(resized_region);
  }

  /**
   * @brief Computes the size of the slide image at a specific scaling factor.
   *
   * This function calculates the dimensions of the slide at a specified scaling
   * level. Optionally, it can limit the calculation to the slide bounds.
   *
   * @param scaling The scaling factor to apply (relative to level 0).
   * @param limit_bounds If true, the scaled size is calculated using the
   * slide's bounded area instead of its full dimensions.
   * @return The dimensions of the scaled image as a Size<int, 2> (width,
   * height).
   */
  [[nodiscard]] Size<int, 2> GetScaledSize(double scaling,
                                           bool limit_bounds = false) const {
    const auto geometry = wsi_->GetScaledGeometry(scaling);
    return limit_bounds ? geometry.bounds : geometry.size;
  }

  /**
   * @brief Retrieves the bounds of the visible tissue region in the slide
   * image.
   *
   * The bounds represent the actual visible region of the slide containing
   * tissue data, which is often smaller than the full slide dimensions. For
   * example, in formats like MRXS, the visible region is explicitly defined and
   * smaller than the full slide size.
   *
   * These bounds help avoid processing empty background areas and focus
   * computation on regions that contain actual tissue. They are usually
   * determined by:
   * - Format-specific metadata (e.g., in MRXS format)
   * - Tissue detection algorithms
   * - Manual annotations
   *
   * The bounds are returned as an offset and size tuple, both defined
   * at level 0 (highest resolution) of the image. A scaling factor can be
   * applied to get the bounds at a different resolution.
   *
   * @return A pair of dimensions, where the first represents the offset (x, y),
   * and the second represents the size (width, height) of the visible tissue
   * region.
   * @see GetDimensions() to get the full slide dimensions.
   */
  [[nodiscard]] std::pair<Size<int, 2>, Size<int, 2>> GetSlideBounds() const {
    return wsi_->GetSlideBounds();
  }

  /**
   * @brief Computes the microns-per-pixel (mpp) for a given scaling level.
   *
   * @param scaling The scaling factor relative to level 0.
   * @return The microns-per-pixel value at the specified scaling.
   */
  [[nodiscard]] double GetMpp(double scaling) const noexcept {
    return avg_native_mpp_ / scaling;
  }

  /**
   * @brief Calculates the scaling factor for a given microns-per-pixel (mpp).
   *
   * @param mpp The desired microns-per-pixel value.
   * @return The scaling factor required to achieve the specified mpp.
   */
  [[nodiscard]] double GetScaling(double mpp) const noexcept {
    if (mpp == 0.0) {
      return 1.0;
    }
    return avg_native_mpp_ / mpp;
  }

  /**
   * @brief Retrieves the average microns-per-pixel (mpp) for the slide image.
   *
   * This value is calculated as the average of the x and y mpp values.
   *
   * @return The average mpp value.
   */
  [[nodiscard]] double GetSpacing() const noexcept { return avg_native_mpp_; }

  /**
   * @brief Finds the closest native level for a given microns-per-pixel (mpp).
   *
   * This function identifies the native resolution level whose average mpp is
   * nearest to the specified mpp value.
   *
   * @param mpp The desired microns-per-pixel value.
   * @return The index of the closest native resolution level.
   */
  [[nodiscard]] int GetClosestNativeLevel(double mpp) const {
    const auto& level_spacings = wsi_->GetLevelSpacings();
    auto min_it = std::ranges::min_element(
        level_spacings, [mpp](const auto& a, const auto& b) {
          const double avg_a = (a[0] + a[1]) / 2.0;
          const double avg_b = (b[0] + b[1]) / 2.0;
          return std::abs(avg_a - mpp) < std::abs(avg_b - mpp);
        });
    return static_cast<int>(
        std::ranges::distance(level_spacings.begin(), min_it));
  }

  /**
   * @brief Retrieves the closest native microns-per-pixel (mpp) value for a
   * given mpp.
   *
   * @param mpp The desired microns-per-pixel value.
   * @return A tuple containing the closest mpp values for x and y axes.
   */
  [[nodiscard]] auto GetClosestNativeMpp(double mpp) const {
    return wsi_->GetLevelSpacings().at(GetClosestNativeLevel(mpp));
  }

  /**
   * @brief Retrieves the identifier for this slide image.
   *
   * The identifier is a user-defined label, often used for debugging or
   * exception messages.
   *
   * @return A string view containing the identifier.
   */
  [[nodiscard]] std::string_view GetIdentifier() const noexcept {
    return identifier_;
  }

  /**
   * @brief Retrieves the properties of the slide image.
   *
   * These properties include additional metadata provided by the backend.
   *
   * @return A vector of strings representing the properties of the slide image.
   */
  [[nodiscard]] std::vector<std::string> GetProperties() const {
    return wsi_->GetProperties();
  }

  /**
   * @brief Retrieves the vendor of the slide scanner.
   *
   * This information can help determine the hardware or software source of the
   * slide.
   *
   * @return An optional string containing the vendor name, or nullopt if
   * unavailable.
   */
  [[nodiscard]] std::optional<std::string> GetVendor() const {
    return wsi_->GetVendor();
  }

  /**
   * @brief Retrieves the resampling method used for interpolation.
   *
   * The interpolation method is used when downscaling or resizing regions.
   *
   * @return The resampling method (e.g., kLanczos or kNearest).
   */
  [[nodiscard]] Resampling GetInterpolator() const noexcept {
    return interpolator_;
  }

  /**
   * @brief Retrieves the full dimensions of the highest resolution level of the
   * slide.
   *
   * @return The dimensions (width, height) of the slide at level 0.
   */
  [[nodiscard]] Size<int, 2> GetDimensions() const {
    return wsi_->GetDimensions();
  }

  [[nodiscard]] double GetMpp() const noexcept { return avg_native_mpp_; }

  /**
   * @brief Retrieves the optical magnification used to acquire the slide.
   *
   * @return An optional double containing the magnification value, or nullopt
   * if unavailable.
   */
  [[nodiscard]] std::optional<double> GetMagnification() const {
    return wsi_->GetMagnification();
  }

  /**
   * @brief Computes the aspect ratio (width/height) of the slide image.
   *
   * @return The aspect ratio as a double.
   */
  [[nodiscard]] double GetAspectRatio() const {
    const auto size = GetDimensions();
    return static_cast<double>(size[0]) / static_cast<double>(size[1]);
  }

  /**
   * @brief Computes the bounds of the visible tissue region in the slide at a
   * specific scaling level.
   *
   * The bounds represent the actual visible region of the slide containing
   * tissue data, which is often smaller than the full slide dimensions. For
   * example, in formats like MRXS, the visible region is explicitly defined and
   * smaller than the full slide size.
   *
   * These bounds are useful for:
   * - Focusing processing on regions containing actual tissue
   * - Avoiding unnecessary computation on empty background areas
   * - Creating properly sized grids for tiling operations
   *
   * The bounds are scaled proportionally to the given scaling factor.
   *
   * @param scaling The scaling factor to apply.
   * @return A pair of scaled bounds, where the first element represents the
   * offset (x, y), and the second element represents the size (width, height)
   * of the visible region.
   */
  [[nodiscard]] std::pair<aifocore::Size<int, 2>, aifocore::Size<int, 2>>
  GetScaledSlideBounds(double scaling) const {
    const auto geometry = wsi_->GetScaledGeometry(scaling);
    return {geometry.offset, geometry.bounds};
  }

  /**
   * @brief Retrieves a thumbnail image for the slide.
   *
   * This function generates a resized image of the slide with a maximum
   * bounding box. The thumbnail is useful for quick visualization of the slide.
   *
   * @param size The maximum dimensions (width, height) for the thumbnail.
   * Defaults to (512, 512).
   * @return A shared pointer to a VImage representing the thumbnail.
   */
  [[nodiscard]] std::shared_ptr<vips::VImage> GetThumbnail(
      const aifocore::Size<int, 2>& size = {512, 512}) const {
    return wsi_->GetThumbnail(size);
  }

  /**
   * @brief Retrieves the full size of the highest resolution level of the
   * slide.
   *
   * This is an alias for GetDimensions() to provide a consistent API.
   *
   * @return The size (width, height) of the slide at level 0.
   */
  [[nodiscard]] Size<int, 2> GetSize() const {
    return wsi_->GetGeometry().size;
  }

  /**
   * @brief Retrieves the size of the slide at a specific scaling level.
   *
   * This is an alias for GetScaledSize() to provide a more consistent naming
   * scheme.
   *
   * @param scaling The scaling factor to apply (relative to level 0).
   * @param limit_bounds If true, the scaled size is calculated using the
   * slide's bounded area instead of its full dimensions.
   * @return The size of the scaled image as a Size<int, 2> (width, height).
   */
  [[nodiscard]] Size<int, 2> GetScaled(double scaling,
                                       bool limit_bounds = false) const {
    return GetScaledSize(scaling, limit_bounds);
  }

  /**
   * @brief Retrieves the geometry of the slide.
   *
   * This returns a SlideGeometry struct containing the size, offset, and bounds
   * of the slide image.
   *
   * @return A SlideGeometry struct.
   */
  [[nodiscard]] SlideGeometry GetGeometry() const {
    return wsi_->GetGeometry();
  }

  /**
   * @brief Retrieves the scaled geometry of the slide.
   *
   * @param scaling The scaling factor to apply.
   * @return A SlideGeometry struct with scaled dimensions.
   */
  [[nodiscard]] SlideGeometry GetScaledGeometry(double scaling) const {
    return wsi_->GetScaledGeometry(scaling);
  }

 private:
  // Actual constructor should be private
  SlideImage(std::shared_ptr<backends::AbstractSlideBackend> wsi,
             std::string identifier, Resampling interpolator,
             double avg_native_mpp)
      : wsi_(std::move(wsi)),
        identifier_(std::move(identifier)),
        interpolator_(interpolator),
        avg_native_mpp_(avg_native_mpp) {}

  std::shared_ptr<backends::AbstractSlideBackend> wsi_;
  std::string identifier_;
  Resampling interpolator_;
  double avg_native_mpp_;
};

}  // namespace dlup

#endif  // AIFO_DLUP_INCLUDE_DLUP_SLIDE_IMAGE_H_
