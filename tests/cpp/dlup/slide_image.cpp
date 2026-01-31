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
#include <gtest/gtest.h>
#include <vips/vips8>

#include <memory>
#include <string>

#include "dlup/slide_image.h"
#include "test_data_file.h"

namespace dlup::test {

class SlideImageTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (VIPS_INIT("")) {
      vips_error_exit(nullptr);
    }
  }

  static void TearDownTestSuite() { vips_shutdown(); }

  std::shared_ptr<backends::AbstractSlideBackend> CreateBackend() {
    std::string filepath = std::string(TEST_DATA_FILE);
    return std::make_shared<backends::VipsSlide>(filepath);
  }
};

TEST_F(SlideImageTest, TestGetInterpolator) {
  auto backend = CreateBackend();
  aifocore::Result<std::unique_ptr<SlideImage>> slide_image_or =
      SlideImage::Create(backend);
  ASSERT_TRUE(slide_image_or.ok()) << slide_image_or.status();

  const auto& slide_image = slide_image_or.value();
  EXPECT_EQ(slide_image->GetInterpolator(), Resampling::kLanczos);
}

TEST_F(SlideImageTest, TestReadRegionValid) {
  auto backend = CreateBackend();
  aifocore::Result<std::unique_ptr<SlideImage>> slide_image_or =
      SlideImage::Create(backend);
  ASSERT_TRUE(slide_image_or.ok()) << slide_image_or.status();

  const auto& slide_image = slide_image_or.value();
  aifocore::Result<std::shared_ptr<vips::VImage>> region_or =
      slide_image->ReadRegion({0, 0}, 1.0, {512, 512});
  ASSERT_TRUE(region_or.ok()) << region_or.status();

  auto region = region_or.value();
  ASSERT_NE(region, nullptr);
  EXPECT_EQ(region->width(), 512);
  EXPECT_EQ(region->height(), 512);
}

TEST_F(SlideImageTest, TestReadRegionInvalidArguments) {
  auto backend = CreateBackend();
  aifocore::Result<std::unique_ptr<SlideImage>> slide_image_or =
      SlideImage::Create(backend);
  ASSERT_TRUE(slide_image_or.ok()) << slide_image_or.status();
  auto& slide_image = slide_image_or.value();

  {
    // Invalid scaling (< 0)
    aifocore::Result<std::shared_ptr<vips::VImage>> result =
        slide_image->ReadRegion({0, 0}, -1.0, {512, 512});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), aifocore::StatusCode::kInvalidArgument);
  }

  {
    // Invalid size with zero width
    aifocore::Result<std::shared_ptr<vips::VImage>> result =
        slide_image->ReadRegion({0, 0}, 1.0, {0, 512});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), aifocore::StatusCode::kInvalidArgument);
  }

  {
    // Invalid size with zero height
    aifocore::Result<std::shared_ptr<vips::VImage>> result =
        slide_image->ReadRegion({0, 0}, 1.0, {512, 0});
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), aifocore::StatusCode::kInvalidArgument);
  }
}

}  // namespace dlup::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return result;
}
