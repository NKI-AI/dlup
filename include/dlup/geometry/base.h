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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BASE_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BASE_H_

#include <algorithm>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <boost/geometry.hpp>

namespace bg = boost::geometry;

using BoostPoint = bg::model::d2::point_xy<double>;
using BoostPolygon = bg::model::polygon<BoostPoint>;
using BoostRing = bg::model::ring<BoostPoint>;

using FieldType = std::variant<std::monostate, int, std::tuple<int, int>,
                               std::tuple<int, int, int>, std::string, bool>;

namespace dlup::geometry {

class BaseGeometry {
 public:
  virtual ~BaseGeometry() = default;
  std::unordered_map<std::string, FieldType> parameters_;

  // Clone must be implemented by the derived classes
  virtual std::shared_ptr<BaseGeometry> Clone() const = 0;

  virtual void SetField(const std::string& name, FieldType value) {
    parameters_[name] = value;
  }

  std::optional<FieldType> GetField(const std::string& name) const {
    if (auto it = parameters_.find(name); it != parameters_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  auto GetFields() const {
    std::vector<std::string> field_names_;
    field_names_.reserve(parameters_.size());
    std::transform(parameters_.begin(), parameters_.end(),
                   std::back_inserter(field_names_),
                   [](const auto& param) { return param.first; });
    return field_names_;
  }

  std::uintptr_t GetPointerId() const {
    return reinterpret_cast<std::uintptr_t>(this);
  }

  virtual std::string ToWkt()
      const = 0;  // Force derived classes to provide the WKT

  template <typename GeometryType>
  std::string ConvertToWkt(const GeometryType& geometry) const {
    std::stringstream ss;
    ss << boost::geometry::wkt(geometry);
    return ss.str();
  }

 protected:
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_BASE_H_
