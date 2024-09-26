#ifndef DLUP_GEOMETRY_EXCEPTIONS_H
#define DLUP_GEOMETRY_EXCEPTIONS_H

#include <stdexcept>
#include <string>

class GeometryError : public std::runtime_error {
  public:
  explicit GeometryError(const std::string &message) : std::runtime_error(message) {}
};

class GeometryNotFoundError : public GeometryError {
  public:
  explicit GeometryNotFoundError(const std::string &message) : GeometryError(message) {}
};

class GeometryCoordinatesError : public GeometryError {
  public:
  explicit GeometryCoordinatesError(const std::string &message) : GeometryError(message) {}
};

class GeometryIntersectionError : public GeometryError {
  public:
  explicit GeometryIntersectionError(const std::string &message) : GeometryError(message) {}
};

class GeometryTransformationError : public GeometryError {
  public:
  explicit GeometryTransformationError(const std::string &message) : GeometryError(message) {}
};

class GeometryFactoryFunctionError : public GeometryError {
  public:
  explicit GeometryFactoryFunctionError(const std::string &message) : GeometryError(message) {}
};

class GeometryInvalidPolygonError : public GeometryError {
  public:
  explicit GeometryInvalidPolygonError(const std::string &message) : GeometryError(message) {}
};

#endif // DLUP_GEOMETRY_EXCEPTIONS_H
