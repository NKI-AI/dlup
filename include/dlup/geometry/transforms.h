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
#ifndef AIFO_DLUP_INCLUDE_DLUP_TRANSFORMS_H_
#define AIFO_DLUP_INCLUDE_DLUP_TRANSFORMS_H_

#include <memory>
#include <tuple>
#include <vector>
#include "dlup/geometry/polygon.h"

namespace dlup {

std::vector<int> GenerateMaskFromAnnotations(
    const std::vector<std::shared_ptr<dlup::geometry::Polygon>>& annotations,
    const std::tuple<int, int>& mask_size, int default_value);

}  // namespace dlup

#endif  // AIFO_DLUP_INCLUDE_DLUP_TRANSFORMS_H_
