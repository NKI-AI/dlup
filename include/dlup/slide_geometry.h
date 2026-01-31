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
#ifndef AIFO_DLUP_INCLUDE_DLUP_SLIDE_GEOMETRY_H_
#define AIFO_DLUP_INCLUDE_DLUP_SLIDE_GEOMETRY_H_

#include <ostream>
#include "aifocore/concepts/numeric.h"

namespace dlup {

using aifocore::Size;

/**
 * @brief A struct that encapsulates the geometric properties of a slide.
 *
 * This struct contains the full slide dimensions, offset, and bounded region
 * size, which are commonly used properties when working with whole-slide
 * images.
 */
struct SlideGeometry {
  Size<int, 2> size;    // Full dimensions of the slide
  Size<int, 2> offset;  // Offset (usually for visible tissue)
  Size<int, 2> bounds;  // Size of the bounded region

  /**
   * @brief Get a scaled version of the geometry.
   *
   * @param scaling The scaling factor to apply.
   * @return A new SlideGeometry with all dimensions scaled accordingly.
   */
  SlideGeometry Scaled(double scaling) const {
    return {static_cast<Size<int, 2>>(size * scaling),
            static_cast<Size<int, 2>>(offset * scaling),
            static_cast<Size<int, 2>>(bounds * scaling)};
  }

  /**
   * @brief Equality operator for SlideGeometry.
   *
   * @param other The other SlideGeometry to compare with.
   * @return True if both SlideGeometry objects are equal.
   */
  bool operator==(const SlideGeometry& other) const {
    return size == other.size && offset == other.offset &&
           bounds == other.bounds;
  }

  /**
   * @brief Inequality operator for SlideGeometry.
   *
   * @param other The other SlideGeometry to compare with.
   * @return True if the SlideGeometry objects are not equal.
   */
  bool operator!=(const SlideGeometry& other) const {
    return !(*this == other);
  }
};

/**
 * @brief Stream insertion operator for SlideGeometry.
 *
 * @param out Output stream.
 * @param geometry The geometry to print.
 * @return The output stream.
 */
inline std::ostream& operator<<(std::ostream& out,
                                const SlideGeometry& geometry) {
  out << "SlideGeometry{size=(" << geometry.size[0] << ", " << geometry.size[1]
      << "), offset=(" << geometry.offset[0] << ", " << geometry.offset[1]
      << "), bounds=(" << geometry.bounds[0] << ", " << geometry.bounds[1]
      << ")}";
  return out;
}

}  // namespace dlup

#endif  // AIFO_DLUP_INCLUDE_DLUP_SLIDE_GEOMETRY_H_
