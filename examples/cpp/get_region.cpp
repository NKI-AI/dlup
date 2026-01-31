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
/*
 * Example program to extract a region from a WSI file using libdlup and VipsSlideBackend.
 *
 * Usage:
 *   get_region filename x y width height [--level=0] [--output=region.jpg] [--apply-color-profile]
 */
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"

#include "dlup/backends/vips.h"

int main(int argc, char* argv[]) {
  std::string filename;
  int x, y, width, height;
  int level = 0;
  std::string output = "region.jpg";
  bool apply_color_profile = false;

  CLI::App app{"Extract region from a whole slide image using dlup"};

  app.add_option("filename", filename, "Input WSI file")->required();
  app.add_option("x", x, "X coordinate")->required();
  app.add_option("y", y, "Y coordinate")->required();
  app.add_option("width", width, "Region width")->required();
  app.add_option("height", height, "Region height")->required();
  app.add_option("--level", level, "Pyramid level")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--output", output, "Output filename");
  app.add_flag("--apply-color-profile", apply_color_profile,
               "Apply color profile");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  try {
    dlup::backends::VipsSlide slide(filename, false, apply_color_profile);

    std::cout << "Extracting region at level " << level << ":\n"
              << "Position: (" << x << ", " << y << ")\n"
              << "Size: " << width << "x" << height << "\n"
              << "From: " << filename << "\n";

    auto region = slide.ReadRegion({x, y}, level, {width, height});
    if (!region.ok()) {
      std::cerr << "Error: " << region.status() << "\n";
      return 1;
    }

    region.value().write_to_file(output.c_str());
    std::cout << "Region saved to: " << output << "\n";

    slide.Close();
  } catch (const std::exception& e) {
    std::cerr << "Unhandled error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
