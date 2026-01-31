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
#ifndef AIFO_DLUP_INCLUDE_DLUP_UTILITIES_FILETYPE_H_
#define AIFO_DLUP_INCLUDE_DLUP_UTILITIES_FILETYPE_H_

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace dlup::utilities {

enum class FileType { kTiff, kXml, kJson, kJpeg, kPng, kUnknown };

struct FileInfo {
  FileType file_type;
  std::unordered_map<std::string, std::string>
      metadata;  // Generalized metadata for all file types
};

namespace detail {
struct FileSignature {
  std::array<uint8_t, 16> signature;
  uint8_t length;
  FileType type;
};

static constexpr std::array<FileSignature, 4> SIGNATURES = {{
    {{0x49, 0x49, 0x2A, 0x00}, 4, FileType::kTiff},  // TIFF LE
    {{0x4D, 0x4D, 0x00, 0x2A}, 4, FileType::kTiff},  // TIFF BE
    {{0xFF, 0xD8, 0xFF}, 3, FileType::kJpeg},        // JPEG
    {{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A},
     8,
     FileType::kPng}  // PNG
}};

[[nodiscard]] inline FileType MatchSignatureScalar(
    const uint8_t* data) noexcept {
  for (const auto& sig : SIGNATURES) {
    if (std::memcmp(data, sig.signature.data(), sig.length) == 0) {
      return sig.type;
    }
  }
  return FileType::kUnknown;
}
}  // namespace detail

[[nodiscard]] inline std::optional<FileInfo> GetXmlInfo(
    const std::string_view content) {
  auto tag_start = content.find('<', content.find("<?xml"));
  if (tag_start == std::string_view::npos)
    return std::nullopt;

  tag_start = content.find('<', tag_start + 1);
  if (tag_start == std::string_view::npos)
    return std::nullopt;

  auto tag_end = content.find('>', tag_start);
  if (tag_end == std::string_view::npos)
    return std::nullopt;

  auto space_pos = content.find(' ', tag_start);
  if (space_pos > tag_end)
    space_pos = tag_end;

  std::string_view root_tag =
      content.substr(tag_start + 1, space_pos - tag_start - 1);

  // Extract version (optional)
  std::string_view xml_version;
  if (auto version_pos = content.find("version=\"");
      version_pos != std::string_view::npos) {
    auto quote_end = content.find('"', version_pos + 9);
    if (quote_end != std::string_view::npos) {
      xml_version =
          content.substr(version_pos + 9, quote_end - (version_pos + 9));
    }
  }

  // Create metadata map
  std::unordered_map<std::string, std::string> metadata;
  metadata["file_type"] = "XML";
  metadata["root_tag"] = std::string(root_tag);
  if (!xml_version.empty()) {
    metadata["version"] = std::string(xml_version);
  }

  return FileInfo{FileType::kXml, std::move(metadata)};
}

[[nodiscard]] inline std::optional<FileInfo> GetJsonInfo(
    const std::string_view content) {
  if (content.empty() || (content[0] != '{' && content[0] != '[')) {
    return std::nullopt;
  }

  // Try to find a "version" key
  std::unordered_map<std::string, std::string> metadata;
  metadata["file_type"] = "JSON";

  return FileInfo{FileType::kJson, std::move(metadata)};
}

[[nodiscard]] inline FileInfo DetectFileType(
    const std::filesystem::path& file_path) {
  if (!std::filesystem::exists(file_path)) {
    return {FileType::kUnknown, {}};
  }

  std::array<uint8_t, 16> buffer = {};
  std::string content;

  {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
      return {FileType::kUnknown, {}};

    // Read binary header
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (file.gcount() == 0)
      return {FileType::kUnknown, {}};

    // Read first 4 KB for text-based formats
    file.seekg(0);
    content.assign(std::istreambuf_iterator<char>(file),
                   std::istreambuf_iterator<char>());
    if (content.size() > 4096)
      content.resize(4096);
  }

  // Check binary formats
  auto type = detail::MatchSignatureScalar(buffer.data());
  if (type != FileType::kUnknown) {
    return {type, {}};
  }

  // Check for XML
  if (auto xml_info = GetXmlInfo(content)) {
    return *xml_info;
  }

  // Check for JSON
  if (auto json_info = GetJsonInfo(content)) {
    return *json_info;
  }

  return {FileType::kUnknown, {}};
}

}  // namespace dlup::utilities

#endif  // AIFO_DLUP_INCLUDE_DLUP_UTILITIES_FILETYPE_H_
