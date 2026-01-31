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
#ifndef AIFO_DLUP_INCLUDE_DLUP_TIFF_EXCEPTIONS_H_
#define AIFO_DLUP_INCLUDE_DLUP_TIFF_EXCEPTIONS_H_

#include <exception>
#include <stdexcept>
#include <string>

class TiffException : public std::runtime_error {
 public:
  explicit TiffException(const std::string& message)
      : std::runtime_error(message) {}
};

class TiffCompressionNotSupportedError : public TiffException {
 public:
  explicit TiffCompressionNotSupportedError(const std::string& message)
      : TiffException("Compression not supported: " + message) {}
};

class TiffOpenException : public TiffException {
 public:
  explicit TiffOpenException(const std::string& message)
      : TiffException("Failed to open TIFF file: " + message) {}
};

class TiffWriteException : public TiffException {
 public:
  explicit TiffWriteException(const std::string& message)
      : TiffException("Failed to write TIFF data: " + message) {}
};

class TiffSetupException : public TiffException {
 public:
  explicit TiffSetupException(const std::string& message)
      : TiffException("Failed to setup TIFF: " + message) {}
};

class TiffReadException : public TiffException {
 public:
  explicit TiffReadException(const std::string& message)
      : TiffException("Failed to read TIFF data: " + message) {}
};

#endif  // AIFO_DLUP_INCLUDE_DLUP_TIFF_EXCEPTIONS_H_
