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
#include <iomanip>
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"

#include "dlup/backends/fastslide.h"

int main(int argc, char* argv[]) {
  std::string filename;

  CLI::App app{"Print basic metadata from a WSI using FastSlide"};

  app.add_option("filename", filename, "Input WSI file")->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  try {
    dlup::backends::FastSlideSlide slide(filename);

    std::cout << "Slide Information for: " << filename << "\n";
    std::string separator(65, '-');
    std::cout << separator << "\n";

    if (auto mag = slide.GetMagnification()) {
      std::cout << "Magnification: " << *mag << "x\n";
    } else {
      std::cout << "Magnification: Not available\n";
    }

    if (auto vendor = slide.GetVendor()) {
      std::cout << "Vendor: " << *vendor << "\n";
    } else {
      std::cout << "Vendor: Not available\n";
    }

    auto format_name =
        std::dynamic_pointer_cast<dlup::backends::FastSlideMetadata>(
            slide.GetMetadata())
            ->format_name;
    std::cout << "Format: " << format_name << "\n";

    const auto [offset, size] = slide.GetSlideBounds();
    const auto [offset_x, offset_y] = offset;
    const auto [size_width, size_height] = size;

    std::cout << "Slide bounds: (" << offset_x << ", " << offset_y << ") to ("
              << size_width << ", " << size_height << ")\n";

    const int level_count = slide.GetLevelCount();
    std::cout << "\nLevel Information (for " << level_count << " levels):\n";
    std::cout << separator << "\n";
    std::cout << std::setw(6) << "Level" << std::setw(12) << "Width"
              << std::setw(12) << "Height" << std::setw(12) << "Downsample"
              << std::setw(22) << "Spacing (μm/pixel)\n";
    std::cout << separator << "\n";

    for (int level = 0; level < level_count; ++level) {
      const auto [width, height] = slide.GetLevelDimensions(level);
      const auto [spacing_x, spacing_y] = slide.GetLevelSpacing(level);
      const auto downsample = slide.GetLevelDownsample(level);
      if (!downsample.ok()) {
        throw std::runtime_error(downsample.status().ToString());
      }

      std::cout << std::fixed << std::setprecision(2) << std::setw(6) << level
                << std::setw(12) << width << std::setw(12) << height
                << std::setw(12) << downsample.value() << std::setw(8)
                << spacing_x << " × " << spacing_y << "\n";
    }

    slide.Close();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
