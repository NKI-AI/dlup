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
#include <filesystem>
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"

#include "dlup/utilities/filetype.h"

int main(int argc, char* argv[]) {
  std::string file_path;

  CLI::App app{"File type detection utility"};

  app.add_option("-f,--file", file_path, "Path to the file to detect its type")
      ->required();

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  // Convert string to filesystem path
  std::filesystem::path path_obj(file_path);
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType(path_obj);

  using FileType = dlup::utilities::FileType;

  switch (file_info.file_type) {
    case FileType::kTiff:
      std::cout << "File type: TIFF\n";
      break;
    case FileType::kXml:
      std::cout << "File type: XML\n";
      if (file_info.metadata.contains("root_tag")) {
        std::cout << "Root tag: " << file_info.metadata.at("root_tag") << "\n";
      }
      if (file_info.metadata.contains("version")) {
        std::cout << "Version: " << file_info.metadata.at("version") << "\n";
      }
      break;
    case FileType::kJson:
      std::cout << "File type: JSON\n";
      break;
    case FileType::kJpeg:
      std::cout << "File type: JPEG\n";
      break;
    case FileType::kPng:
      std::cout << "File type: PNG\n";
      break;
    default:
      std::cout << "File type: Unknown\n";
  }

  return 0;
}
