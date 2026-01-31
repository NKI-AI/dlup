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
//
// This implementation is based on the marching squares algorithm from
// scikit-image, which is licensed under the BSD-3-Clause license:
// https://github.com/scikit-image/scikit-image
// Copyright (C) 2019, the scikit-image team
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include "dlup/geometry/marching_squares.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dlup/geometry/polygon.h"

namespace dlup::geometry {

namespace {

using Point = std::pair<double, double>;
using Segment = std::pair<Point, Point>;
using Contour = std::vector<Point>;

/**
 * @brief Linear interpolation to find the exact position where the level
 * crosses an edge.
 *
 * @param from_value Value at the start of the edge.
 * @param to_value Value at the end of the edge.
 * @param level The iso-value level.
 * @return Fraction along the edge where the level is crossed (0.0 to 1.0).
 */
inline double GetFraction(double from_value, double to_value, double level) {
  if (to_value == from_value) {
    return 0.0;
  }
  return (level - from_value) / (to_value - from_value);
}

/**
 * @brief Safely get pixel value, returning 0.0 for out-of-bounds access.
 *
 * This implements logical padding without actual memory allocation.
 *
 * @param image Input image as a 2D NDArrayView.
 * @param row Row index.
 * @param col Column index.
 * @param height Image height.
 * @param width Image width.
 * @return Pixel value or 0.0 if out of bounds.
 */
inline double GetPixelSafe(const aifocore::math::NDArrayView<double, 2>& image,
                           int row, int col, int height, int width) {
  if (row < 0 || row >= height || col < 0 || col >= width) {
    return 0.0;  // Logical padding with zeros
  }
  return image(row, col);
}

/**
 * @brief Extracts contour segments using the marching squares algorithm.
 *
 * Uses logical 1-pixel padding (returns 0 for out-of-bounds) to naturally
 * close contours at image boundaries, similar to OpenCV's approach.
 *
 * @param image Input image as a 2D NDArrayView.
 * @param level The iso-value level to extract contours at.
 * @return Vector of line segments (each segment is a pair of points).
 */
std::vector<Segment> GetContourSegments(
    const aifocore::math::NDArrayView<double, 2>& image, double level) {
  std::vector<Segment> segments;

  auto shape = image.Shape();
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);

  // Iterate with logical 1-pixel padding: from -1 to height (inclusive)
  // This treats out-of-bounds pixels as 0, naturally closing border contours
  for (int r0 = -1; r0 < height; ++r0) {
    for (int c0 = -1; c0 < width; ++c0) {
      const int r1 = r0 + 1;
      const int c1 = c0 + 1;

      // Get the four corner values (returns 0.0 for out-of-bounds)
      const double ul = GetPixelSafe(image, r0, c0, height, width);
      const double ur = GetPixelSafe(image, r0, c1, height, width);
      const double ll = GetPixelSafe(image, r1, c0, height, width);
      const double lr = GetPixelSafe(image, r1, c1, height, width);

      // Skip if any value is NaN
      if (std::isnan(ul) || std::isnan(ur) || std::isnan(ll) ||
          std::isnan(lr)) {
        continue;
      }

      // Compute the square case (4-bit value)
      unsigned char square_case = 0;
      if (ul > level)
        square_case += 1;
      if (ur > level)
        square_case += 2;
      if (ll > level)
        square_case += 4;
      if (lr > level)
        square_case += 8;

      // Cases 0 and 15: entirely below/above the contour
      if (square_case == 0 || square_case == 15) {
        continue;
      }

      // Compute interpolated intersection points on edges
      // No offset needed - logical padding is internal only
      const Point top = {static_cast<double>(r0),
                         c0 + GetFraction(ul, ur, level)};
      const Point bottom = {static_cast<double>(r1),
                            c0 + GetFraction(ll, lr, level)};
      const Point left = {r0 + GetFraction(ul, ll, level),
                          static_cast<double>(c0)};
      const Point right = {r0 + GetFraction(ur, lr, level),
                           static_cast<double>(c1)};

      // Generate segments based on the square case
      // Using low-value connectivity (vertex_connect_high = false)
      switch (square_case) {
        case 1:  // top to left
          segments.push_back({top, left});
          break;
        case 2:  // right to top
          segments.push_back({right, top});
          break;
        case 3:  // right to left
          segments.push_back({right, left});
          break;
        case 4:  // left to bottom
          segments.push_back({left, bottom});
          break;
        case 5:  // top to bottom
          segments.push_back({top, bottom});
          break;
        case 6:  // ambiguous case (low-value connectivity)
          segments.push_back({right, top});
          segments.push_back({left, bottom});
          break;
        case 7:  // right to bottom
          segments.push_back({right, bottom});
          break;
        case 8:  // bottom to right
          segments.push_back({bottom, right});
          break;
        case 9:  // ambiguous case (low-value connectivity)
          segments.push_back({top, left});
          segments.push_back({bottom, right});
          break;
        case 10:  // bottom to top
          segments.push_back({bottom, top});
          break;
        case 11:  // bottom to left
          segments.push_back({bottom, left});
          break;
        case 12:  // left to right
          segments.push_back({left, right});
          break;
        case 13:  // top to right
          segments.push_back({top, right});
          break;
        case 14:  // left to top
          segments.push_back({left, top});
          break;
      }
    }
  }

  return segments;
}

/**
 * @brief Assembles disconnected segments into complete contours.
 *
 * @param segments Vector of line segments to assemble.
 * @return Vector of contours (each contour is a vector of points).
 */
std::vector<Contour> AssembleContours(const std::vector<Segment>& segments) {
  std::map<int, std::deque<Point>> contours;
  std::map<Point, std::pair<std::deque<Point>*, int>> starts;
  std::map<Point, std::pair<std::deque<Point>*, int>> ends;

  int current_index = 0;

  for (const auto& [from_point, to_point] : segments) {
    // Ignore degenerate segments
    if (from_point == to_point) {
      continue;
    }

    // Check if we can connect to existing contours
    auto tail_it = starts.find(to_point);
    auto head_it = ends.find(from_point);

    std::deque<Point>* tail = nullptr;
    int tail_num = -1;
    std::deque<Point>* head = nullptr;
    int head_num = -1;

    if (tail_it != starts.end()) {
      tail = tail_it->second.first;
      tail_num = tail_it->second.second;
      starts.erase(tail_it);
    }

    if (head_it != ends.end()) {
      head = head_it->second.first;
      head_num = head_it->second.second;
      ends.erase(head_it);
    }

    if (tail != nullptr && head != nullptr) {
      // Connect two contours or close a loop
      if (tail == head) {
        // Close the contour
        head->push_back(to_point);
      } else {
        // Merge two distinct contours
        if (tail_num > head_num) {
          // Append tail to head
          head->insert(head->end(), tail->begin(), tail->end());
          contours.erase(tail_num);
          starts[head->front()] = {head, head_num};
          ends[head->back()] = {head, head_num};
        } else {
          // Prepend head to tail (in reverse)
          for (auto it = head->rbegin(); it != head->rend(); ++it) {
            tail->push_front(*it);
          }
          starts.erase(head->front());
          contours.erase(head_num);
          starts[tail->front()] = {tail, tail_num};
          ends[tail->back()] = {tail, tail_num};
        }
      }
    } else if (tail == nullptr && head == nullptr) {
      // Create a new contour
      std::deque<Point> new_contour = {from_point, to_point};
      contours[current_index] = new_contour;
      std::deque<Point>* contour_ptr = &contours[current_index];
      starts[from_point] = {contour_ptr, current_index};
      ends[to_point] = {contour_ptr, current_index};
      current_index++;
    } else if (head == nullptr) {
      // Prepend to tail
      tail->push_front(from_point);
      starts[from_point] = {tail, tail_num};
    } else {  // tail == nullptr
      // Append to head
      head->push_back(to_point);
      ends[to_point] = {head, head_num};
    }
  }

  // Convert deques to vectors and sort by index
  std::vector<Contour> result;
  for (const auto& [idx, contour_deque] : contours) {
    result.emplace_back(contour_deque.begin(), contour_deque.end());
  }

  return result;
}

}  // namespace

std::vector<std::shared_ptr<Polygon>> FindContours(
    const aifocore::math::NDArrayView<double, 2>& image, double level) {
  auto shape = image.Shape();
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);

  if (height < 2 || width < 2) {
    throw std::invalid_argument("Input array must be at least 2x2.");
  }

  // Extract segments using marching squares (with logical padding)
  std::vector<Segment> segments = GetContourSegments(image, level);

  // Assemble segments into contours
  // Contours are automatically closed due to logical padding
  std::vector<Contour> contours = AssembleContours(segments);

  // Convert contours to Polygon objects
  std::vector<std::shared_ptr<Polygon>> result;
  for (auto& contour : contours) {
    if (contour.empty()) {
      continue;
    }
    result.push_back(std::make_shared<Polygon>(
        contour, std::vector<std::vector<std::pair<double, double>>>()));
  }

  return result;
}

}  // namespace dlup::geometry