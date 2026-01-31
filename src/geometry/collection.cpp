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
#include "dlup/geometry/collection.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dlup::geometry {

GeometryCollection::GeometryCollection() : rtree_wrapper_(this) {}

std::shared_ptr<GeometryCollection> GeometryCollection::Clone(
    bool deepcopy) const {
  std::lock_guard<std::mutex> lock(collection_mutex_);

  // Create a new instance of GeometryCollection
  auto new_collection = std::make_shared<GeometryCollection>();

  if (deepcopy) {
    // Deep copy each component
    for (const auto& polygon : polygons_) {
      new_collection->polygons_.emplace_back(
          std::make_shared<Polygon>(*polygon));
    }
    for (const auto& roi : rois_) {
      new_collection->rois_.emplace_back(std::make_shared<Polygon>(*roi));
    }
    for (const auto& point : points_) {
      new_collection->points_.emplace_back(std::make_shared<Point>(*point));
    }
    for (const auto& box : boxes_) {
      new_collection->boxes_.emplace_back(std::make_shared<Box>(*box));
    }
  } else {
    // Shallow copy each component
    new_collection->polygons_ = polygons_;
    new_collection->rois_ = rois_;
    new_collection->points_ = points_;
    new_collection->boxes_ = boxes_;
  }

  if (!rtree_wrapper_.IsInvalidated()) {
    new_collection->RebuildRTree();
  }
  return new_collection;
}

std::pair<std::pair<double, double>, std::pair<double, double>>
GeometryCollection::ComputeBoundingBox() const {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  BoostBox overall_bounding_box_;
  bool is_first_ = true;

  // Iterate over all polygons and compute their bounding boxes
  for (const auto& polygon : polygons_) {
    BoostBox polygon_box;
    bg::envelope(*(polygon->polygon_), polygon_box);

    if (is_first_) {
      overall_bounding_box_ = polygon_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, polygon_box);
    }
  }

  // Iterate over the ROIs and compute their bounding boxes
  for (const auto& roi : rois_) {
    BoostBox roi_box;
    bg::envelope(*(roi->polygon_), roi_box);

    if (is_first_) {
      overall_bounding_box_ = roi_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, roi_box);
    }
  }

  // Iterate over all boxes and compute their bounding boxes
  for (const auto& box : boxes_) {
    if (is_first_) {
      overall_bounding_box_ = *(box->box_);
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, *(box->box_));
    }
  }

  // Iterate over all points and compute their bounding boxes
  for (const auto& point : points_) {
    BoostBox point_box(*(point->point_), *(point->point_));

    if (is_first_) {
      overall_bounding_box_ = point_box;
      is_first_ = false;
    } else {
      bg::expand(overall_bounding_box_, point_box);
    }
  }

  // Extract min and max points
  const auto& min_corner = overall_bounding_box_.min_corner();
  const auto& max_corner = overall_bounding_box_.max_corner();

  double min_x = bg::get<0>(min_corner);
  double min_y = bg::get<1>(min_corner);
  double max_x = bg::get<0>(max_corner);
  double max_y = bg::get<1>(max_corner);

  double width = max_x - min_x;
  double height = max_y - min_y;

  return std::make_pair(std::make_pair(min_x, min_y),
                        std::make_pair(width, height));
}

void GeometryCollection::ReindexPolygons(
    const std::map<std::string, int>& index_map) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto& polygon : polygons_) {
    std::optional<FieldType> label_opt = polygon->GetField("label");

    if (label_opt.has_value()) {
      // Use std::get_if to safely access the string inside the variant
      const std::string* label_ptr = std::get_if<std::string>(&(*label_opt));
      if (label_ptr) {
        std::string label = *label_ptr;
        auto it = index_map.find(label);
        if (it != index_map.end()) {
          // Safely set the "index" field
          polygon->SetField("index", it->second);
        } else {
          throw std::invalid_argument(
              aifocore::fmt::format("Label {} not found in index_map", label));
        }
      } else {
        throw std::invalid_argument("The 'label' field is not a string.");
      }
    } else {
      throw std::invalid_argument(
          "Polygon does not have a value for the 'label' field");
    }
  }
}

std::unordered_map<std::string, int> GeometryCollection::GetIndexMap() const {
  std::lock_guard<std::mutex> lock(collection_mutex_);  // Ensure thread safety
  std::unordered_map<std::string, int> index_map;

  for (const auto& polygon : polygons_) {
    // Safely retrieve the label and index fields
    std::optional<FieldType> label_opt = polygon->GetField("label");
    std::optional<FieldType> index_opt = polygon->GetField("index");

    // Continue only if both fields exist
    if (label_opt.has_value() && index_opt.has_value()) {
      // Safely extract the values using std::get_if
      const std::string* label_ptr = std::get_if<std::string>(&(*label_opt));
      const int* index_ptr = std::get_if<int>(&(*index_opt));

      if (label_ptr && index_ptr) {
        // Add the label-index pair to the map
        index_map[*label_ptr] = *index_ptr;
      } else {
        // Log or handle the case where types are mismatched
        // For now, we simply skip the entry
      }
    }
  }

  return index_map;
}

void RTreeWrapper::Rebuild() {
  Clear();  // Clear the existing R-tree

  // First insert polygons
  const auto& polygons = geometryCollection->polygons_;
  for (size_t i = 0; i < polygons.size(); ++i) {
    BoostBox box;
    bg::envelope(*(polygons[i]->polygon_), box);
    Insert(box, i);
  }

  // Next insert ROIs
  const auto& rois = geometryCollection->rois_;
  for (size_t i = 0; i < rois.size(); ++i) {
    BoostBox box;
    bg::envelope(*(rois[i]->polygon_), box);
    Insert(box, polygons.size() + i);
  }

  // Next insert boxes
  const auto& boxes = geometryCollection->boxes_;
  for (size_t i = 0; i < boxes.size(); ++i) {
    Insert(*(boxes[i]->box_), polygons.size() + rois.size() + i);
  }

  // Finally, insert points
  const auto& points = geometryCollection->points_;
  for (size_t i = 0; i < points.size(); ++i) {
    BoostBox box(*(points[i]->point_), *(points[i]->point_));
    Insert(box, polygons.size() + rois.size() + boxes.size() + i);
  }

  rtree_invalidated_ = false;
}

void GeometryCollection::AddPolygon(const PolygonPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  polygons_.emplace_back(p);
  rtree_wrapper_.Invalidate();
  num_polygons_++;
}

void GeometryCollection::AddRoi(const PolygonPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  rois_.emplace_back(p);
  rtree_wrapper_.Invalidate();
  num_rois_++;
}

void GeometryCollection::AddPoint(const PointPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  points_.emplace_back(p);
  rtree_wrapper_.Invalidate();
  num_points_++;
}

void GeometryCollection::AddBox(const BoxPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  boxes_.emplace_back(p);
  rtree_wrapper_.Invalidate();
  num_boxes_++;
}

void GeometryCollection::SortPolygons(
    const std::function<bool(const PolygonPtr&, const PolygonPtr&)>&
        comparator) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  std::sort(polygons_.begin(), polygons_.end(), comparator);
  rtree_wrapper_.Invalidate();
}

void GeometryCollection::Scale(double scaling) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto& point : points_) {
    point->Scale(scaling);
  }
  for (auto& polygon : polygons_) {
    polygon->Scale(scaling);
  }

  for (auto& roi : rois_) {
    roi->Scale(scaling);
  }

  for (auto& box : boxes_) {
    box->Scale(scaling);
  }
  rtree_wrapper_.Invalidate();
}

void GeometryCollection::SetOffset(std::pair<double, double> offset) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  for (auto& point : points_) {
    dlup::geometry::utilities::AffineTransform(
        *point->point_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto& polygon : polygons_) {
    dlup::geometry::utilities::AffineTransform(
        *polygon->polygon_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto& roi : rois_) {
    dlup::geometry::utilities::AffineTransform(
        *roi->polygon_, {-offset.first, -offset.second}, 1.0);
  }
  for (auto& box : boxes_) {
    dlup::geometry::utilities::AffineTransform(
        *box->box_, {-offset.first, -offset.second}, 1.0);
  }

  rtree_wrapper_.Invalidate();
}

void GeometryCollection::RemovePolygon(const PolygonPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(polygons_.begin(), polygons_.end(), p);
  if (it != polygons_.end()) {
    polygons_.erase(it);
    rtree_wrapper_.Invalidate();
    num_polygons_--;
  } else {
    throw GeometryNotFoundError("Polygon not found");
  }
}

void GeometryCollection::RemovePolygon(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= polygons_.size()) {
    throw std::out_of_range("Polygon index out of range");
  }

  polygons_.erase(polygons_.begin() + index);
  rtree_wrapper_.Invalidate();
  num_polygons_--;
}

void GeometryCollection::RemoveBox(const BoxPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(boxes_.begin(), boxes_.end(), p);
  if (it != boxes_.end()) {
    boxes_.erase(it);
    rtree_wrapper_.Invalidate();
    num_boxes_--;
  } else {
    throw GeometryNotFoundError("Box not found");
  }
}

void GeometryCollection::RemoveBox(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= boxes_.size()) {
    throw std::out_of_range("Box index out of range");
  }

  boxes_.erase(boxes_.begin() + index);
  rtree_wrapper_.Invalidate();
  num_boxes_--;
}

void GeometryCollection::RemoveRoi(const PolygonPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(rois_.begin(), rois_.end(), p);
  if (it != rois_.end()) {
    rois_.erase(it);
    rtree_wrapper_.Invalidate();
  } else {
    throw GeometryNotFoundError("ROI not found");
  }
  num_rois_--;
}

void GeometryCollection::RemoveRoi(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= rois_.size()) {
    throw std::out_of_range("ROI index out of range");
  }

  rois_.erase(rois_.begin() + index);
  rtree_wrapper_.Invalidate();
  num_rois_--;
}

void GeometryCollection::RemovePoint(const PointPtr& p) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  auto it = std::find(points_.begin(), points_.end(), p);
  if (it != points_.end()) {
    points_.erase(it);
    rtree_wrapper_.Invalidate();
    num_points_--;
  } else {
    throw GeometryNotFoundError("Point not found");
  }
}

void GeometryCollection::RemovePoint(size_t index) {
  std::lock_guard<std::mutex> lock(collection_mutex_);
  if (index >= points_.size()) {
    throw std::out_of_range("Point index out of range");
  }

  points_.erase(points_.begin() + index);
  rtree_wrapper_.Invalidate();
  num_points_--;
}

AnnotationRegion GeometryCollection::ReadRegion(
    const std::pair<double, double>& coordinates, double scaling,
    const std::pair<double, double>& size) const {
  return AnnotationRegion(
      [=, this]() {
        std::lock_guard<std::mutex> lock(collection_mutex_);
        if (rtree_wrapper_.IsInvalidated()) {
          throw std::runtime_error(
              "R-tree is invalidated. Please rebuild before accessing "
              "regions.");
        }

        BoostPoint top_left(coordinates.first / scaling,
                            coordinates.second / scaling);
        BoostPoint bottom_right((coordinates.first + size.first) / scaling,
                                (coordinates.second + size.second) / scaling);
        BoostBox query_box(top_left, bottom_right);

        BoostPolygon intersection_polygon;
        bg::convert(query_box, intersection_polygon);
        std::vector<std::pair<BoostBox, size_t>> results;
        rtree_wrapper_.Query(bgi::intersects(query_box),
                             std::back_inserter(results));

        std::sort(
            results.begin(), results.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        std::vector<std::shared_ptr<Polygon>> intersected_polygons;
        std::vector<std::shared_ptr<Polygon>> intersected_rois;
        std::vector<std::shared_ptr<Point>> current_points;
        std::vector<std::shared_ptr<Box>> current_boxes;

        for (const auto& result : results) {
          size_t index = result.second;
          if (index < polygons_.size()) {
            auto& polygon = polygons_[index];
            auto intersections = polygon->Intersection(intersection_polygon);
            for (const auto& intersected_polygon : intersections) {
              dlup::geometry::utilities::AffineTransform(
                  *intersected_polygon->polygon_, coordinates, scaling);
              intersected_polygons.push_back(intersected_polygon);
            }
          } else if (index < polygons_.size() + rois_.size()) {
            auto& roi = rois_[index - polygons_.size()];
            auto intersection = roi->Intersection(intersection_polygon);
            for (const auto& intersected_roi : intersection) {
              dlup::geometry::utilities::AffineTransform(
                  *intersected_roi->polygon_, coordinates, scaling);
              intersected_rois.push_back(intersected_roi);
            }
          } else if (index < polygons_.size() + rois_.size() + boxes_.size()) {
            auto& box = boxes_[index - polygons_.size() - rois_.size()];
            auto transformed_box = std::make_shared<Box>(*box);
            dlup::geometry::utilities::AffineTransform(*transformed_box->box_,
                                                       coordinates, scaling);
            current_boxes.push_back(transformed_box);
          } else {
            auto& point = points_[index - polygons_.size() - rois_.size() -
                                  boxes_.size()];
            auto transformed_point = std::make_shared<Point>(*point);

            dlup::geometry::utilities::AffineTransform(
                *transformed_point->point_, coordinates, scaling);
            current_points.push_back(transformed_point);
          }
        }

        return AnnotationRegion(
            std::move(intersected_polygons), std::move(intersected_rois),
            std::move(current_boxes), std::move(current_points),
            std::make_tuple(size.first, size.second));
      },
      HasRois());
}

}  // namespace dlup::geometry
