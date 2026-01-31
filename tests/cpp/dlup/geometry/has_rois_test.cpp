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

#include <memory>
#include <vector>

#include "dlup/geometry/collection.h"
#include "dlup/geometry/polygon.h"
#include "dlup/geometry/region.h"

namespace dlup::geometry::test {

class GeometryCollectionHasRoisTest : public ::testing::Test {
 protected:
  // Set up some test polygons
  std::shared_ptr<Polygon> CreateSquarePolygon() {
    auto polygon = std::make_shared<Polygon>();
    polygon->SetExterior({{0, 0}, {0, 100}, {100, 100}, {100, 0}});
    return polygon;
  }
};

// Test HasRois returns false when collection has no ROIs
TEST_F(GeometryCollectionHasRoisTest, NoRois) {
  GeometryCollection collection;

  // Add a regular polygon (not an ROI)
  collection.AddPolygon(CreateSquarePolygon());

  // HasRois should return false
  EXPECT_FALSE(collection.HasRois());
  EXPECT_EQ(collection.NumRois(), 0);
}

// Test HasRois returns true when collection has ROIs
TEST_F(GeometryCollectionHasRoisTest, HasRois) {
  GeometryCollection collection;

  // Add an ROI
  collection.AddRoi(CreateSquarePolygon());

  // HasRois should return true
  EXPECT_TRUE(collection.HasRois());
  EXPECT_EQ(collection.NumRois(), 1);
}

// Test that HasRois is properly passed to AnnotationRegion during ReadRegion
TEST_F(GeometryCollectionHasRoisTest, ReadRegionWithRois) {
  GeometryCollection collection;

  // Add an ROI that will be within our read region
  auto roi = CreateSquarePolygon();
  collection.AddRoi(roi);

  // Make sure the R-tree is built
  collection.RebuildRTree();

  // Read a region that will include the ROI (note that HasRois should be true
  // regardless of whether the region includes the ROI)
  auto region = collection.ReadRegion({0, 0}, 1.0, {100, 100});

  // The region should have its has_rois property set to true
  EXPECT_TRUE(region.HasRois());
}

// Test that HasRois is properly passed to AnnotationRegion
// during ReadRegion when no ROIs
TEST_F(GeometryCollectionHasRoisTest, ReadRegionWithoutRois) {
  GeometryCollection collection;

  // Add a regular polygon (not an ROI)
  collection.AddPolygon(CreateSquarePolygon());

  // Make sure the R-tree is built
  collection.RebuildRTree();

  // Read a region
  auto region = collection.ReadRegion({0, 0}, 1.0, {100, 100});

  // The region should have its has_rois property set to false
  EXPECT_FALSE(region.HasRois());
}

// Test that has_rois is correctly set in the constructor
TEST_F(GeometryCollectionHasRoisTest, AnnotationRegionHasRoisImmutable) {
  // Create AnnotationRegion with has_rois set to true
  AnnotationRegion region(std::vector<std::shared_ptr<Polygon>>{},
                          std::vector<std::shared_ptr<Polygon>>{},
                          std::vector<std::shared_ptr<Box>>{},
                          std::vector<std::shared_ptr<Point>>{},
                          std::make_tuple(100, 100), true);

  // Verify has_rois is true
  EXPECT_TRUE(region.HasRois());

  // Create another region with has_rois set to false
  AnnotationRegion region2(std::vector<std::shared_ptr<Polygon>>{},
                           std::vector<std::shared_ptr<Polygon>>{},
                           std::vector<std::shared_ptr<Box>>{},
                           std::vector<std::shared_ptr<Point>>{},
                           std::make_tuple(100, 100), false);

  // Verify has_rois is false
  EXPECT_FALSE(region2.HasRois());
}

// Test that has_rois reflects the parent collection state
// even if the region doesn't include any ROIs
TEST_F(GeometryCollectionHasRoisTest, ReadRegionOutsideRois) {
  GeometryCollection collection;

  // Add a regular polygon
  collection.AddPolygon(CreateSquarePolygon());

  // Add an ROI that will be outside our read region
  auto roi = std::make_shared<Polygon>();
  roi->SetExterior({{200, 200}, {200, 300}, {300, 300}, {300, 200}});
  collection.AddRoi(roi);

  // The collection should have ROIs
  EXPECT_TRUE(collection.HasRois());

  // Make sure the R-tree is built
  collection.RebuildRTree();

  // Read a region that doesn't include any ROIs
  auto region = collection.ReadRegion({0, 0}, 1.0, {100, 100});

  // Even though the region doesn't contain ROIs, has_rois should still be true
  // because it reflects the state of the parent collection
  EXPECT_TRUE(region.HasRois());
}

}  // namespace dlup::geometry::test

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
