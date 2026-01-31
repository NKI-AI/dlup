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
#include "dlup/backends/abstract.h"

#include <gtest/gtest.h>
#include <vips/vips8>

#include <memory>
#include <string>
#include <vector>

#include "aifocore/concepts/numeric.h"
#include "aifocore/status/result.h"
#include "dlup/slide_geometry.h"

using aifocore::Size;

namespace dlup::backends::test {

class MockSlideBackend : public AbstractSlideBackend {
 public:
  explicit MockSlideBackend() : AbstractSlideBackend("test_filename.tiff") {
    metadata_->level_count = 3;
    metadata_->level_dimensions = {
        {1000, 1000},  // level 0
        {500, 500},    // level 1
        {250, 250}     // level 2
    };
    metadata_->level_downsamples = {1.0, 2.0, 4.0};
    SetSpacing({0.5, 0.5});  // This will populate level_spacings_
    metadata_->slide_bounds = {{100, 100},
                               {800, 800}};  // Set offset and bounds
    metadata_->slide_geometry =
        dlup::SlideGeometry{{1000, 1000}, {100, 100}, {800, 800}};
  }

  std::optional<double> GetMagnification() const override {
    return std::nullopt;
  }

  std::optional<std::string> GetVendor() const override { return std::nullopt; }

  std::vector<std::string> GetProperties() const override { return {}; }

  aifocore::Result<vips::VImage> ReadRegion(
      const aifocore::Size<int, 2>&, int,
      const aifocore::Size<int, 2>&) const override {
    return vips::VImage();
  }

  void Close() override {}
};

TEST(TestMockSlideBackend, TestLevelProperties) {
  MockSlideBackend slide;

  // Test the level count
  EXPECT_EQ(slide.GetLevelCount(), 3);

  // Test base dimensions
  const auto [base_width, base_height] = slide.GetDimensions();
  EXPECT_EQ(base_width, 1000);
  EXPECT_EQ(base_height, 1000);

  // Test level dimensions
  const auto [width0, height0] = slide.GetLevelDimensions(0);
  EXPECT_EQ(width0, 1000);
  EXPECT_EQ(height0, 1000);

  const auto [width1, height1] = slide.GetLevelDimensions(1);
  EXPECT_EQ(width1, 500);
  EXPECT_EQ(height1, 500);

  const auto [width2, height2] = slide.GetLevelDimensions(2);
  EXPECT_EQ(width2, 250);
  EXPECT_EQ(height2, 250);

  // Test invalid level dimensions
  EXPECT_THROW(static_cast<void>(slide.GetLevelDimensions(-1)),
               std::out_of_range);
  EXPECT_THROW(static_cast<void>(slide.GetLevelDimensions(3)),
               std::out_of_range);
}

TEST(TestMockSlideBackend, TestSpacings) {
  MockSlideBackend slide;

  // Test the level spacings
  const auto [spacing0_x, spacing0_y] = slide.GetLevelSpacing(0);
  EXPECT_DOUBLE_EQ(spacing0_x, 0.5);
  EXPECT_DOUBLE_EQ(spacing0_y, 0.5);

  const auto [spacing1_x, spacing1_y] = slide.GetLevelSpacing(1);
  EXPECT_DOUBLE_EQ(spacing1_x, 1.0);
  EXPECT_DOUBLE_EQ(spacing1_y, 1.0);

  const auto [spacing2_x, spacing2_y] = slide.GetLevelSpacing(2);
  EXPECT_DOUBLE_EQ(spacing2_x, 2.0);
  EXPECT_DOUBLE_EQ(spacing2_y, 2.0);

  // Test setting invalid spacings
  auto status = slide.SetSpacing({0.0, 0.5});
  EXPECT_FALSE(status.ok());
  // EXPECT_EQ(status.message(), "Cannot set mpp values.");

  status = slide.SetSpacing({0.5, 0.0});
  EXPECT_FALSE(status.ok());
  // EXPECT_EQ(status.message(), "Cannot set mpp values.");

  status = slide.SetSpacing({0.5, 0.5});
  EXPECT_TRUE(status.ok());
  // EXPECT_EQ(status.message(), "");

  status = slide.SetSpacing({-0.5, 0.5});
  EXPECT_FALSE(status.ok());
  // EXPECT_EQ(status.message(), "Cannot set mpp values.");

  status = slide.SetSpacing({0.5, -0.5});
  EXPECT_FALSE(status.ok());
  // EXPECT_EQ(status.message(), "Cannot set mpp values.");
}

TEST(TestMockSlideBackend, TestDownsamples) {
  MockSlideBackend slide;

  // Test the level downsamples
  aifocore::Result<double> downsample0 = slide.GetLevelDownsample(0);
  EXPECT_TRUE(downsample0.ok());
  EXPECT_DOUBLE_EQ(downsample0.value(), 1.0);

  aifocore::Result<double> downsample1 = slide.GetLevelDownsample(1);
  EXPECT_TRUE(downsample1.ok());
  EXPECT_DOUBLE_EQ(downsample1.value(), 2.0);

  aifocore::Result<double> downsample2 = slide.GetLevelDownsample(2);
  EXPECT_TRUE(downsample2.ok());
  EXPECT_DOUBLE_EQ(downsample2.value(), 4.0);
}

TEST(TestMockSlideBackend, TestBestDownsampleLevel) {
  MockSlideBackend slide;

  // Test the best level for downsample
  EXPECT_EQ(slide.GetBestLevelForDownsample(0.5), 0);
  EXPECT_EQ(slide.GetBestLevelForDownsample(1.0), 0);
  EXPECT_EQ(slide.GetBestLevelForDownsample(1.5), 0);
  EXPECT_EQ(slide.GetBestLevelForDownsample(2.0), 1);
  EXPECT_EQ(slide.GetBestLevelForDownsample(2.5), 1);
  EXPECT_EQ(slide.GetBestLevelForDownsample(4.0), 2);
  EXPECT_EQ(slide.GetBestLevelForDownsample(5.0), 2);
}

TEST(TestMockSlideBackend, TestFilename) {
  MockSlideBackend slide;
  EXPECT_EQ(slide.GetFilename(), "test_filename.tiff");
}

TEST(TestMockSlideBackend, TestSlideGeometry) {
  MockSlideBackend slide;

  // Test the GetGeometry method
  auto geometry = slide.GetGeometry();
  EXPECT_EQ(geometry.size, (Size<int, 2>{1000, 1000}));
  EXPECT_EQ(geometry.offset, (Size<int, 2>{100, 100}));
  EXPECT_EQ(geometry.bounds, (Size<int, 2>{800, 800}));

  // Test the GetScaledGeometry method
  auto scaled_geometry = slide.GetScaledGeometry(0.5);
  EXPECT_EQ(scaled_geometry.size, (Size<int, 2>{500, 500}));
  EXPECT_EQ(scaled_geometry.offset, (Size<int, 2>{50, 50}));
  EXPECT_EQ(scaled_geometry.bounds, (Size<int, 2>{400, 400}));

  // Test the Scaled method of SlideGeometry directly
  // Create a local instance to avoid namespace issues
  dlup::SlideGeometry test_geometry{
      Size<int, 2>{1000, 1000}, Size<int, 2>{100, 100}, Size<int, 2>{800, 800}};
  auto scaled_again = test_geometry.Scaled(0.25);
  EXPECT_EQ(scaled_again.size, (Size<int, 2>{250, 250}));
  EXPECT_EQ(scaled_again.offset, (Size<int, 2>{25, 25}));
  EXPECT_EQ(scaled_again.bounds, (Size<int, 2>{200, 200}));
}

}  // namespace dlup::backends::test

int main(int argc, char** argv) {
  // Initialize VIPS
  if (VIPS_INIT(argv[0])) {
    vips_error_exit(NULL);
    return 1;
  }
  vips_concurrency_set(1);
  vips_cache_set_max(0);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  // Shutdown VIPS
  vips_shutdown();
  return result;
}
