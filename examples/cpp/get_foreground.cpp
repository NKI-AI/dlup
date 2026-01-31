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
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "CLI11/CLI11.hpp"

#include "aifocore/concepts/numeric.h"
#include "aifocore/status/result.h"
#include "aifocore/tiling/grid.h"
#include "dlup/backends/vips.h"
#include "dlup/foreground.h"

namespace tiling = aifocore::tiling;
namespace fs = std::filesystem;

aifocore::Result<std::shared_ptr<vips::VImage>> GetTile(
    dlup::SlideImage* image, const aifocore::Size<int, 2>& coordinates,
    aifocore::Size<int, 2> tile_size, double scaling, int scaled_width,
    int scaled_height) {
  tile_size[0] = std::min(tile_size[0], scaled_width - coordinates[0]);
  tile_size[1] = std::min(tile_size[1], scaled_height - coordinates[1]);

  std::shared_ptr<vips::VImage> region;
  AIFOCORE_ASSIGN_OR_RETURN(
      region,
      image->ReadRegion(static_cast<aifocore::Size<double, 2>>(coordinates),
                        scaling, tile_size));
  return region;
}

int main(int argc, char* argv[]) {
  std::string image_file;
  std::string mask_file;
  int tile_size = 0;
  double threshold = 0.0;
  std::string output_folder = "./output/";
  double mpp = 1.0;

  CLI::App app{"Extract foreground tiles from WSI and mask using libdlup"};

  app.add_option("--image", image_file, "Input WSI image file")->required();
  app.add_option("--mask", mask_file, "Input mask file")->required();
  app.add_option("--tile_size", tile_size, "Tile size in pixels")->required();
  app.add_option("--threshold", threshold, "Foreground threshold");
  app.add_option("--output", output_folder, "Output folder");
  app.add_option("--mpp", mpp, "Microns-per-pixel value");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (tile_size <= 0) {
    std::cerr << "Error: --tile_size must be > 0\n";
    return 1;
  }

  fs::path output_path(output_folder);
  if (!fs::exists(output_path)) {
    fs::create_directories(output_path);
  }

  auto slide_backend = std::make_shared<dlup::backends::VipsSlide>(image_file);
  aifocore::Result<std::unique_ptr<dlup::SlideImage>> image_result =
      dlup::SlideImage::Create(slide_backend);
  if (!image_result.ok()) {
    std::cerr << "Failed to create SlideImage: " << image_result.status()
              << "\n";
    return 1;
  }
  std::unique_ptr<dlup::SlideImage>& image = image_result.value();

  auto mask_backend = std::make_shared<dlup::backends::VipsSlide>(mask_file);

  const double scaling = image->GetScaling(mpp);
  const auto [start_coordinates, scaled_size] =
      image->GetScaledSlideBounds(scaling);

  tiling::Grid<int> grid = tiling::Grid<int>::FromTiling(
      {0, 0}, scaled_size, {tile_size, tile_size}, {0, 0},
      tiling::TilingMode::kSkip, tiling::GridOrder::kC);

  std::cout << "Unfiltered grid length: " << grid.Length() << "\n";

  const auto start = std::chrono::steady_clock::now();
  const auto foreground_result = dlup::Foreground<int>::FilterGrid(
      grid, *mask_backend, {tile_size, tile_size}, mpp, threshold);
  const auto end = std::chrono::steady_clock::now();

  std::cout << "FilterGrid took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms.\n";

  std::cout << "Found " << foreground_result.foreground_indices.size()
            << " foreground tiles from " << grid.Length() << " total tiles\n";

  size_t tile_count = 0;
  for (int idx : foreground_result.foreground_indices) {
    const auto coord = grid[idx];
    aifocore::Result<std::shared_ptr<vips::VImage>> region =
        GetTile(image.get(), {coord[0], coord[1]}, {tile_size, tile_size},
                scaling, scaled_size[0], scaled_size[1]);

    if (!region.ok()) {
      std::cerr << "Tile read error: " << region.status() << "\n";
      return 1;
    }

    auto processed = region.value()->colourspace(VIPS_INTERPRETATION_sRGB);

    const std::string filename =
        (output_path /
         fs::path("tile_" + std::to_string(tile_count++) + ".jpg"))
            .string();
    processed.jpegsave(filename.c_str(), vips::VImage::option()
                                             ->set("Q", 90)
                                             ->set("strip", true)
                                             ->set("optimize_coding", true));
  }

  std::cout << "Extracted " << tile_count << " foreground tiles to "
            << output_path << "\n";
  image->Close();

  return 0;
}
