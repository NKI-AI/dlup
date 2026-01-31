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
 * Example program to extract a scaled region from a WSI file using SlideImage.
 * Usage:
 *   get_scaled_region filename x y width height --mpp <μm/px> [--output=region.jpg]
 */
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "CLI11/CLI11.hpp"

#include "dlup/backends/vips.h"
#include "dlup/slide_image.h"

int main(int argc, char* argv[]) {
  std::string filename;
  double x, y;
  int width, height;
  double mpp;
  std::string output = "region.jpg";

  CLI::App app{"Extract a scaled region from a WSI using libdlup::SlideImage"};

  app.add_option("filename", filename, "Input WSI file")->required();
  app.add_option("x", x, "X coordinate")->required();
  app.add_option("y", y, "Y coordinate")->required();
  app.add_option("width", width, "Region width")->required();
  app.add_option("height", height, "Region height")->required();
  app.add_option("--mpp", mpp, "Microns per pixel")->required();
  app.add_option("--output", output, "Output filename");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  try {
    auto slide_backend = std::make_shared<dlup::backends::VipsSlide>(filename);

    aifocore::Result<std::unique_ptr<dlup::SlideImage>> maybe_slide_image =
        dlup::SlideImage::Create(slide_backend);

    if (!maybe_slide_image.ok()) {
      std::cerr << "Error: " << maybe_slide_image.status() << "\n";
      return 1;
    }

    std::unique_ptr<dlup::SlideImage> slide_image =
        std::move(maybe_slide_image.value());

    double scaling = slide_image->GetScaling(mpp);
    int closest_level = slide_image->GetClosestNativeLevel(mpp);

    std::cout << "Extracting region:\n"
              << "Position: (" << x << ", " << y << ")\n"
              << "Size: " << width << "x" << height << "\n"
              << "Scaling: " << scaling << "\n"
              << "Closest level: " << closest_level << "\n"
              << "From: " << filename << "\n";

    auto region = slide_image->ReadRegion({x, y}, scaling, {width, height});
    if (!region.ok()) {
      std::cerr << "Error: " << region.status() << "\n";
      return 1;
    }

    region.value()->write_to_file(output.c_str());
    std::cout << "Region saved to: " << output << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
