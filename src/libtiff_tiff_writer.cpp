#include "constants.h"
#include "image.h"
#include "tiff/exceptions.h"
#include "tiff/writer.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdexcept>
#include <string>
#include <tiffio.h>
#include <vector>

PYBIND11_MODULE(_libtiff_tiff_writer, m) {
  py::class_<LibtiffTiffWriter>(m, "LibtiffTiffWriter")
      .def(py::init([](py::object path, std::array<int, 3> Size, std::array<float, 2> mpp, std::array<int, 2> tileSize,
                       py::object compression, int quality) {
        fs::path cpp_path;
        if (py::isinstance<py::str>(path)) {
          cpp_path = fs::path(path.cast<std::string>());
        } else if (py::hasattr(path, "__fspath__")) {
          cpp_path = fs::path(path.attr("__fspath__")().cast<std::string>());
        } else {
          throw py::type_error("Expected str or os.PathLike object");
        }

        CompressionType comp_type;
        if (py::isinstance<py::str>(compression)) {
          comp_type = string_to_compression_type(compression.cast<std::string>());
        } else if (py::isinstance<CompressionType>(compression)) {
          comp_type = compression.cast<CompressionType>();
        } else {
          throw py::type_error("Expected str or CompressionType for compression");
        }

        return new LibtiffTiffWriter(std::move(cpp_path), Size, mpp, tileSize, comp_type, quality);
      }))
      .def("write_tile", &LibtiffTiffWriter::writeTile)
      .def("write_pyramid", &LibtiffTiffWriter::writePyramid)
      .def("finalize", &LibtiffTiffWriter::finalize);

  py::enum_<CompressionType>(m, "CompressionType")
      .value("NONE", CompressionType::NONE)
      .value("JPEG", CompressionType::JPEG)
      .value("LZW", CompressionType::LZW)
      .value("DEFLATE", CompressionType::DEFLATE);

  py::register_exception<TiffException>(m, "TiffException");
  py::register_exception<TiffOpenException>(m, "TiffOpenException");
  py::register_exception<TiffReadException>(m, "TiffReadException");
  py::register_exception<TiffWriteException>(m, "TiffWriteException");
  py::register_exception<TiffSetupException>(m, "TiffSetupException");
  py::register_exception<TiffCompressionNotSupportedError>(m, "TiffCompressionNotSupportedError");
}
