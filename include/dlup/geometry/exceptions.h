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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_EXCEPTIONS_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace dlup::geometry {

class GeometryError : public std::runtime_error {
 public:
  explicit GeometryError(const std::string& message)
      : std::runtime_error(message) {}
};

class GeometryNotFoundError : public GeometryError {
 public:
  explicit GeometryNotFoundError(const std::string& message)
      : GeometryError(message) {}
};

class GeometryCoordinatesError : public GeometryError {
 public:
  explicit GeometryCoordinatesError(const std::string& message)
      : GeometryError(message) {}
};

class GeometryIntersectionError : public GeometryError {
 public:
  explicit GeometryIntersectionError(const std::string& message)
      : GeometryError(message) {}
};

class GeometryTransformationError : public GeometryError {
 public:
  explicit GeometryTransformationError(const std::string& message)
      : GeometryError(message) {}
};

class GeometryFactoryFunctionError : public GeometryError {
 public:
  explicit GeometryFactoryFunctionError(const std::string& message)
      : GeometryError(message) {}
};

class GeometryInvalidPolygonError : public GeometryError {
 public:
  explicit GeometryInvalidPolygonError(const std::string& message)
      : GeometryError(message) {}
};

}  // namespace dlup::geometry

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_EXCEPTIONS_H_
