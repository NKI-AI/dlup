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
#ifndef AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_LAZY_ARRAY_H_
#define AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_LAZY_ARRAY_H_

#include <functional>
#include <utility>
#include <vector>

template <typename T>
class LazyArray {
 public:
  using ComputeFunction = std::function<std::vector<T>()>;

  LazyArray(ComputeFunction compute_func, std::vector<std::size_t> shape)
      : compute_func_(std::move(compute_func)),
        computed_(false),
        shape_(std::move(shape)) {}

  // Return the computed data as a vector
  const std::vector<T>& data() const {
    if (!computed_) {
      data_ = compute_func_();  // Compute lazily
      computed_ = true;
    }
    return data_;
  }

  // Return the shape
  const std::vector<std::size_t>& shape() const { return shape_; }

 private:
  ComputeFunction compute_func_;    // Function to compute the data
  mutable std::vector<T> data_;     // Store the computed data
  mutable bool computed_;           // Flag to indicate if computed
  std::vector<std::size_t> shape_;  // Shape of the array
};

#endif  // AIFO_DLUP_INCLUDE_DLUP_GEOMETRY_LAZY_ARRAY_H_
