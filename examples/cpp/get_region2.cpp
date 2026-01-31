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
 * Example program to extract a region from a WSI file using libdlup and VipsSlideBackend,
 * and reinitialize a second instance using shared metadata.
 *
 * Usage:
 *   get_region2 filename x y width height [--level=0] [--output=region.jpg]
 */
#include <iostream>
#include <memory>
#include <string>

#include "CLI11/CLI11.hpp"

#include "dlup/backends/vips.h"

int main(int argc, char* argv[]) {
  std::string filename;
  int x, y, width, height;
  int level = 0;
  std::string output = "region.jpg";

  CLI::App app{"Extract region from a WSI using shared metadata in libdlup"};

  app.add_option("filename", filename, "Input WSI file")->required();
  app.add_option("x", x, "X coordinate")->required();
  app.add_option("y", y, "Y coordinate")->required();
  app.add_option("width", width, "Region width")->required();
  app.add_option("height", height, "Region height")->required();
  app.add_option("--level", level, "Pyramid level")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--output", output, "Output filename");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  try {
    std::cout << "Loading slide and metadata from: " << filename << "\n";
    dlup::backends::VipsSlide slide(filename);

    auto vips_metadata =
        std::dynamic_pointer_cast<dlup::backends::VipsSlideMetadata>(
            slide.GetMetadata());

    std::cout << "Metadata loaded:\n";
    std::cout << "Filename: " << vips_metadata->filename << "\n";
    std::cout << "Loader: " << vips_metadata->loader << "\n";
    std::cout << "Levels: " << vips_metadata->level_count << "\n";

    std::cout << "\nReinitializing slide from shared metadata...\n";
    dlup::backends::VipsSlide slide_from_metadata(*vips_metadata);

    std::cout << "Extracting region at level " << level << ":\n"
              << "Position: (" << x << ", " << y << ")\n"
              << "Size: " << width << "x" << height << "\n";

    auto region =
        slide_from_metadata.ReadRegion({x, y}, level, {width, height});
    if (!region.ok()) {
      std::cerr << "Error: " << region.status() << "\n";
      return 1;
    }

    region.value().write_to_file(output.c_str());
    std::cout << "Region saved to: " << output << "\n";

    slide_from_metadata.Close();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
