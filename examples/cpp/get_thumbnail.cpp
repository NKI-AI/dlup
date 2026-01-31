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
 * Example program to generate a thumbnail from a WSI using libdlup's VipsSlide backend.
 *
 * Usage:
 *   get_thumbnail filename width height [--output=thumbnail.jpg] [--apply-color-profile]
 */

#define CLI11_HAS_EXCEPTIONS 0

#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"

#include "dlup/backends/vips.h"

int main(int argc, char* argv[]) {
  std::string filename;
  int width, height;
  std::string output = "thumbnail.jpg";
  bool apply_color_profile = false;

  CLI::App app{"Generate thumbnail from WSI using libdlup"};

  app.add_option("filename", filename, "Input WSI file")->required();
  app.add_option("width", width, "Thumbnail width")->required();
  app.add_option("height", height, "Thumbnail height")->required();
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

    std::cout << "Generating thumbnail of size " << width << "x" << height
              << " from " << filename << "\n";

    auto thumbnail = slide.GetThumbnail({width, height});
    thumbnail->write_to_file(output.c_str());

    std::cout << "Thumbnail saved to: " << output << "\n";
    slide.Close();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
