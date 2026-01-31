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
#include "dlup/slide_geometry.h"
#include <gtest/gtest.h>

// Test the SlideGeometry struct directly from header
TEST(SlideGeometryTest, BasicProperties) {
  // Create a test geometry
  aifocore::Size<int, 2> size{1000, 800};
  aifocore::Size<int, 2> offset{100, 100};
  aifocore::Size<int, 2> bounds{800, 600};

  dlup::SlideGeometry geometry{size, offset, bounds};

  // Test properties
  EXPECT_EQ(geometry.size[0], 1000);
  EXPECT_EQ(geometry.size[1], 800);
  EXPECT_EQ(geometry.offset[0], 100);
  EXPECT_EQ(geometry.offset[1], 100);
  EXPECT_EQ(geometry.bounds[0], 800);
  EXPECT_EQ(geometry.bounds[1], 600);
}

TEST(SlideGeometryTest, Scaling) {
  // Create a test geometry
  aifocore::Size<int, 2> size{1000, 800};
  aifocore::Size<int, 2> offset{100, 100};
  aifocore::Size<int, 2> bounds{800, 600};

  dlup::SlideGeometry geometry{size, offset, bounds};

  // Test scaling by 0.5
  dlup::SlideGeometry scaled = geometry.Scaled(0.5);
  EXPECT_EQ(scaled.size[0], 500);
  EXPECT_EQ(scaled.size[1], 400);
  EXPECT_EQ(scaled.offset[0], 50);
  EXPECT_EQ(scaled.offset[1], 50);
  EXPECT_EQ(scaled.bounds[0], 400);
  EXPECT_EQ(scaled.bounds[1], 300);

  // Test scaling by 2.0
  scaled = geometry.Scaled(2.0);
  EXPECT_EQ(scaled.size[0], 2000);
  EXPECT_EQ(scaled.size[1], 1600);
  EXPECT_EQ(scaled.offset[0], 200);
  EXPECT_EQ(scaled.offset[1], 200);
  EXPECT_EQ(scaled.bounds[0], 1600);
  EXPECT_EQ(scaled.bounds[1], 1200);
}

TEST(SlideGeometryTest, ChainedScaling) {
  // Create a test geometry
  aifocore::Size<int, 2> size{1000, 800};
  aifocore::Size<int, 2> offset{100, 100};
  aifocore::Size<int, 2> bounds{800, 600};

  dlup::SlideGeometry geometry{size, offset, bounds};

  // Test chained scaling (0.5 * 0.5 = 0.25)
  dlup::SlideGeometry scaled1 = geometry.Scaled(0.5);
  dlup::SlideGeometry scaled2 = scaled1.Scaled(0.5);

  // Compare with direct scaling by 0.25
  dlup::SlideGeometry direct = geometry.Scaled(0.25);

  EXPECT_EQ(scaled2.size[0], direct.size[0]);
  EXPECT_EQ(scaled2.size[1], direct.size[1]);
  EXPECT_EQ(scaled2.offset[0], direct.offset[0]);
  EXPECT_EQ(scaled2.offset[1], direct.offset[1]);
  EXPECT_EQ(scaled2.bounds[0], direct.bounds[0]);
  EXPECT_EQ(scaled2.bounds[1], direct.bounds[1]);

  // Original geometry should be unchanged
  EXPECT_EQ(geometry.size[0], 1000);
  EXPECT_EQ(geometry.size[1], 800);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
