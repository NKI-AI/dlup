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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BOX_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BOX_H_

#include <memory>
#include <string>
#include <vector>

#include <boost/geometry.hpp>

#include "dlup/geometry/base.h"
#include "dlup/geometry/exceptions.h"
#include "dlup/geometry/polygon.h"
#include "dlup/geometry/utilities.h"

namespace dlup::geometry {

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostBox = bg::model::box<BoostPoint>;

class Box : public BaseGeometry {
 public:
  ~Box() override = default;
  std::shared_ptr<BoostBox> box_;

  Box();
  explicit Box(const BoostBox& p);
  explicit Box(std::shared_ptr<BoostBox> p);
  Box(const std::array<double, 2>& coordinates,
      const std::array<double, 2>& size);

  std::shared_ptr<BaseGeometry> Clone() const override;
  void SetBoxParameters(const std::array<double, 2>& coordinates,
                        const std::array<double, 2>& size);
  std::array<double, 2> GetCoordinates() const;
  std::array<double, 2> GetSize() const;
  double GetArea() const;
  std::shared_ptr<Polygon> AsPolygon() const;
  std::vector<std::pair<double, double>> GetExterior() const;

  void Scale(double scaling);
  static std::shared_ptr<Box> create(std::array<double, 2> coordinates,
                                     std::array<double, 2> size);

  std::string ToWkt() const override { return ConvertToWkt(*box_); }
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BOX_H_
