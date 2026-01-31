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
#include "dlup/utilities/filetype.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

// Helper function to write test files
void CreateTestFile(const std::string& filename, const std::string& content) {
  std::ofstream file(filename, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << "Failed to create test file: " << filename;
  file << content;
  file.close();
}

// Test Fixture
class FileTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create test files in the temporary directory
    CreateTestFile("test_tiff.tiff",
                   "\x49\x49\x2A\x00");  // Little-endian TIFF header
    CreateTestFile("test_jpeg.jpg", "\xFF\xD8\xFF");            // JPEG header
    CreateTestFile("test_png.png", "\x89PNG\x0D\x0A\x1A\x0A");  // PNG header

    CreateTestFile("test_dlup.xml",
                   R"(<?xml version="1.0" encoding="UTF-8"?>
                   <DlupAnnotations version="1.0">
                   </DlupAnnotations>)");
    CreateTestFile("test_asap.xml",
                   R"(<?xml version="1.0"?>
                   <ASAP_Annotations>
                   </ASAP_Annotations>)");
    CreateTestFile("test_unknown.xml",
                   R"(<?xml version="1.0"?>
                   <UnknownTag>
                   </UnknownTag>)");
  }

  void TearDown() override {
    // Clean up test files
    std::remove("test_tiff.tiff");
    std::remove("test_jpeg.jpg");
    std::remove("test_png.png");
    std::remove("test_dlup.xml");
    std::remove("test_asap.xml");
    std::remove("test_unknown.xml");
  }
};

// Test for TIFF file detection
TEST_F(FileTypeTest, DetectTiff) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_tiff.tiff");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kTiff);
  EXPECT_TRUE(file_info.metadata.empty());
}

// Test for JPEG file detection
TEST_F(FileTypeTest, DetectJpeg) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_jpeg.jpg");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kJpeg);
  EXPECT_TRUE(file_info.metadata.empty());
}

// Test for PNG file detection
TEST_F(FileTypeTest, DetectPng) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_png.png");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kPng);
  EXPECT_TRUE(file_info.metadata.empty());
}

// Test for DlupAnnotations XML detection
TEST_F(FileTypeTest, DetectDlupXml) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_dlup.xml");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kXml);
  EXPECT_EQ(file_info.metadata.at("root_tag"), "DlupAnnotations");
  EXPECT_EQ(file_info.metadata.at("version"), "1.0");
}

// Test for ASAP_Annotations XML detection
TEST_F(FileTypeTest, DetectAsapXml) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_asap.xml");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kXml);
  EXPECT_EQ(file_info.metadata.at("root_tag"), "ASAP_Annotations");
}

// Test for unknown XML root tag detection
TEST_F(FileTypeTest, DetectUnknownXml) {
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_unknown.xml");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kXml);
  EXPECT_EQ(file_info.metadata.at("root_tag"), "UnknownTag");
}

// Test for unknown file type detection
TEST_F(FileTypeTest, DetectUnknownFile) {
  CreateTestFile("test_unknown.bin", "\x00\x01\x02\x03\x04\x05");
  dlup::utilities::FileInfo file_info =
      dlup::utilities::DetectFileType("test_unknown.bin");
  EXPECT_EQ(file_info.file_type, dlup::utilities::FileType::kUnknown);
  EXPECT_TRUE(file_info.metadata.empty());
  std::remove("test_unknown.bin");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
