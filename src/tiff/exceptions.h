#ifndef DLUP_TIFF_EXCEPTIONS_H
#define DLUP_TIFF_EXCEPTIONS_H
#pragma once

#include <exception>
#include <stdexcept>
#include <string>

class TiffException : public std::runtime_error {
  public:
  explicit TiffException(const std::string &message) : std::runtime_error(message) {}
};

class TiffCompressionNotSupportedError : public TiffException {
  public:
  explicit TiffCompressionNotSupportedError(const std::string &message)
      : TiffException("Compression not supported: " + message) {}
};

class TiffOpenException : public TiffException {
  public:
  explicit TiffOpenException(const std::string &message) : TiffException("Failed to open TIFF file: " + message) {}
};

class TiffWriteException : public TiffException {
  public:
  explicit TiffWriteException(const std::string &message) : TiffException("Failed to write TIFF data: " + message) {}
};

class TiffSetupException : public TiffException {
  public:
  explicit TiffSetupException(const std::string &message) : TiffException("Failed to setup TIFF: " + message) {}
};

class TiffReadException : public TiffException {
  public:
  explicit TiffReadException(const std::string &message) : TiffException("Failed to read TIFF data: " + message) {}
};

#endif // DLUP_TIFF_EXCEPTIONS_H
