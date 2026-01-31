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
#include "dlup/backends/vips.h"

#include <gtest/gtest.h>
#include <vips/vips8>

#include <memory>
#include <string>

#include "aifocore/concepts/numeric.h"
#include "test_data_file.h"

namespace dlup::backends::test {

class VipsSlideTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (VIPS_INIT("")) {
      vips_error_exit(nullptr);
    }
  }

  static void TearDownTestSuite() { vips_shutdown(); }

  std::shared_ptr<VipsSlide> CreateVipsSlide() {
    std::string filepath = std::string(TEST_DATA_FILE);
    return std::make_shared<VipsSlide>(filepath);
  }
};

TEST_F(VipsSlideTest, TestMetadataLoading) {
  auto vips_slide = CreateVipsSlide();

  // Check basic metadata
  EXPECT_EQ(vips_slide->GetLevelCount(), 4);  // 4 levels in checkerboard.svs
  EXPECT_EQ(vips_slide->GetDimensions(), (Dimensions{16000, 16000}));
  EXPECT_EQ(vips_slide->GetLevelDimensions(0), (Dimensions{16000, 16000}));
  EXPECT_EQ(vips_slide->GetLevelDimensions(1), (Dimensions{4000, 4000}));
  EXPECT_EQ(vips_slide->GetLevelDimensions(2), (Dimensions{1000, 1000}));
  EXPECT_EQ(vips_slide->GetLevelDimensions(3), (Dimensions{250, 250}));

  EXPECT_TRUE(vips_slide->GetMagnification().has_value());
  EXPECT_DOUBLE_EQ(vips_slide->GetMagnification().value(), 40.0);

  auto vendor = vips_slide->GetVendor();
  ASSERT_TRUE(vendor.has_value());
  EXPECT_EQ(vendor.value(), "aperio");
}

TEST_F(VipsSlideTest, TestSlideBounds) {
  auto vips_slide = CreateVipsSlide();

  auto bounds = vips_slide->GetSlideBounds();
  EXPECT_EQ(bounds.first, (Dimensions{0, 0}));           // Offset
  EXPECT_EQ(bounds.second, (Dimensions{16000, 16000}));  // Size
}

TEST_F(VipsSlideTest, TestReadRegionValid) {
  auto vips_slide = CreateVipsSlide();

  // Read a region from level 0
  aifocore::Size<int, 2> region_coords = {0, 0};
  aifocore::Size<int, 2> region_size = {512, 512};

  auto region = vips_slide->ReadRegion(region_coords, 0, region_size);
  EXPECT_TRUE(region.ok());
  EXPECT_EQ(region.value().width(), 512);
  EXPECT_EQ(region.value().height(), 512);
}

TEST_F(VipsSlideTest, TestReadRegionOutOfBounds) {
  auto vips_slide = CreateVipsSlide();

  // Request a region outside the bounds
  aifocore::Size<int, 2> region_coords = {16500, 16500};
  aifocore::Size<int, 2> region_size = {512, 512};

  auto test_read = [&]() {
    auto region = vips_slide->ReadRegion(region_coords, 0, region_size);
    return region;
  };
  auto status = test_read();
  EXPECT_FALSE(status.ok());
  // EXPECT_EQ(status.message(), "Invalid size or location");
}

TEST_F(VipsSlideTest, TestLevelDownsampling) {
  auto vips_slide = CreateVipsSlide();

  for (int level = 0; level < vips_slide->GetLevelCount(); ++level) {
    aifocore::Result<double> downsample = vips_slide->GetLevelDownsample(level);
    EXPECT_TRUE(downsample.ok());
    if (level == 0) {
      EXPECT_DOUBLE_EQ(downsample.value(), 1.0);  // Native resolution
    } else {
      EXPECT_GT(downsample.value(), 1.0);  // Downsampled
    }
  }
}

TEST_F(VipsSlideTest, TestProperties) {
  auto vips_slide = CreateVipsSlide();

  auto properties = vips_slide->GetProperties();
  EXPECT_FALSE(properties.empty());

  // Check for a specific property
  EXPECT_NE(
      std::find(properties.begin(), properties.end(), "openslide.comment"),
      properties.end());
}

TEST_F(VipsSlideTest, TestThumbnail) {
  auto vips_slide = CreateVipsSlide();

  // Generate a thumbnail
  aifocore::Size<int, 2> thumbnail_size = {256, 256};
  auto thumbnail = vips_slide->GetThumbnail(thumbnail_size);

  ASSERT_NE(thumbnail, nullptr);
  EXPECT_EQ(thumbnail->width(), 256);
  EXPECT_EQ(thumbnail->height(), 256);
}

}  // namespace dlup::backends::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
