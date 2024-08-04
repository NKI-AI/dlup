#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tiffio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <array>
#include <filesystem>

namespace fs = std::filesystem;
namespace py = pybind11;

class TiffException : public std::runtime_error {
public:
    explicit TiffException(const std::string& message) : std::runtime_error(message) {}
};

class TiffOpenException : public TiffException {
public:
    explicit TiffOpenException(const std::string& message) : TiffException("Failed to open TIFF file: " + message) {}
};

class TiffWriteException : public TiffException {
public:
    explicit TiffWriteException(const std::string& message) : TiffException("Failed to write TIFF data: " + message) {}
};

class TiffSetupException : public TiffException {
public:
    explicit TiffSetupException(const std::string& message) : TiffException("Failed to setup TIFF: " + message) {}
};

enum class CompressionType {
    NONE,
    JPEG,
    LZW,
    DEFLATE
};

CompressionType string_to_compression_type(const std::string& compression) {
    if (compression == "NONE") return CompressionType::NONE;
    if (compression == "JPEG") return CompressionType::JPEG;
    if (compression == "LZW") return CompressionType::LZW;
    if (compression == "DEFLATE") return CompressionType::DEFLATE;
    throw std::invalid_argument("Invalid compression type: " + compression);
}

struct TIFFDeleter {
    void operator()(TIFF* tif) const noexcept {
        if (tif) {
            // Disable error reporting temporarily
            TIFFErrorHandler oldHandler = TIFFSetErrorHandler(nullptr);

            // Attempt to flush any pending writes
            if (TIFFFlush(tif) == 0) {
                TIFFError("TIFFDeleter", "Failed to flush TIFF data");
            }

            TIFFClose(tif);
            TIFFSetErrorHandler(oldHandler);
        }
    }
};

using TIFFPtr = std::unique_ptr<TIFF, TIFFDeleter>;

class FastTiffWriter {
public:
    FastTiffWriter(fs::path filename,
                   std::array<int, 3> size,
                   double mpp,
                   std::array<int, 2> tile_size = {512, 512},
                   CompressionType compression = CompressionType::JPEG,
                   int quality = 100)
        : filename(std::move(filename)), size(size), mpp(mpp), tile_size(tile_size),
          compression(compression), quality(quality),
          tif(nullptr)
    {
        validateInputs();

        TIFF* tiff_ptr = TIFFOpen(this->filename.c_str(), "w");
        if (!tiff_ptr) {
            throw TiffOpenException("Unable to create TIFF file");
        }
        tif.reset(tiff_ptr);

        setupTIFF();
    }

    void write_tile(py::array_t<std::byte, py::array::c_style | py::array::forcecast> tile, int row, int col) {
        auto buf = tile.request();
        if (buf.ndim < 2 || buf.ndim > 3) {
            throw TiffWriteException("Invalid number of dimensions in tile data. Expected 2 or 3, got " + std::to_string(buf.ndim));
        }
        auto [height, width, channels] = std::tuple{buf.shape[0], buf.shape[1], buf.ndim > 2 ? buf.shape[2] : 1};

        // Verify dimensions and buffer size
        size_t expected_size = static_cast<size_t>(width) * height * channels;
        if (static_cast<size_t>(buf.size) != expected_size) {
            throw TiffWriteException("Buffer size does not match expected size. Expected " + std::to_string(expected_size) + ", got " + std::to_string(buf.size));
        }

        // Check if tile coordinates are within bounds
        if (row < 0 || row >= size[0] || col < 0 || col >= size[1]) {
            throw TiffWriteException("Tile coordinates out of bounds for row " + std::to_string(row) + ", col " + std::to_string(col) + ". Image size is " + std::to_string(size[0]) + "x" + std::to_string(size[1]));
        }

        // Write the tile
        if (TIFFWriteTile(tif.get(), buf.ptr, col, row, 0, 0) < 0) {
            throw TiffWriteException("TIFFWriteTile failed for row " + std::to_string(row) + ", col " + std::to_string(col));
        }
    }

private:
    std::string filename;
    std::array<int, 3> size;
    double mpp;
    std::array<int, 2> tile_size;
    CompressionType compression;
    int quality;
    TIFFPtr tif;

    void validateInputs() const {
        if (size[0] <= 0 || size[1] <= 0 || size[2] <= 0) {
            throw std::invalid_argument("Invalid size parameters");
        }
        if (mpp <= 0) {
            throw std::invalid_argument("Invalid mpp value");
        }
        if (tile_size[0] <= 0 || tile_size[1] <= 0) {
            throw std::invalid_argument("Invalid tile size");
        }
        if (quality < 0 || quality > 100) {
            throw std::invalid_argument("Invalid quality value");
        }
    }

    void setupTIFF() {
        int width = size[1];
        int height = size[0];
        int channels = size[2];

        auto set_field = [this](uint32_t tag, auto... value) {
            if (TIFFSetField(tif.get(), tag, value...) != 1) {
                throw TiffSetupException("Failed to set TIFF field: " + std::to_string(tag));
            }
        };

        set_field(TIFFTAG_IMAGEWIDTH, width);
        set_field(TIFFTAG_IMAGELENGTH, height);
        set_field(TIFFTAG_SAMPLESPERPIXEL, channels);
        set_field(TIFFTAG_BITSPERSAMPLE, 8);
        set_field(TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        set_field(TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        set_field(TIFFTAG_TILEWIDTH, tile_size[1]);
        set_field(TIFFTAG_TILELENGTH, tile_size[0]);

        if (channels == 3 || channels == 4) {
            set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        } else {
            set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        }

        if (channels == 4) {
            uint16_t extra_samples = EXTRASAMPLE_ASSOCALPHA;
            set_field(TIFFTAG_EXTRASAMPLES, 1, &extra_samples);
        } else if (channels > 4) {
            std::vector<uint16_t> extra_samples(channels - 3, EXTRASAMPLE_UNSPECIFIED);
            set_field(TIFFTAG_EXTRASAMPLES, channels - 3, extra_samples.data());
        }

        switch (compression) {
            case CompressionType::NONE:
                set_field(TIFFTAG_COMPRESSION, COMPRESSION_NONE);
                break;
            case CompressionType::JPEG:
                set_field(TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
                set_field(TIFFTAG_JPEGQUALITY, quality);
                set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_YCBCR);
                set_field(TIFFTAG_YCBCRSUBSAMPLING, 2, 2);
                set_field(TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
                break;
            case CompressionType::LZW:
                set_field(TIFFTAG_COMPRESSION, COMPRESSION_LZW);
                break;
            case CompressionType::DEFLATE:
                set_field(TIFFTAG_COMPRESSION, COMPRESSION_ADOBE_DEFLATE);
                break;
            default:
                throw TiffSetupException("Unknown compression type");
        }

        // Convert mpp (micrometers per pixel) to pixels per centimeter
        double pixels_per_cm = 10000.0 / mpp;

        set_field(TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
        set_field(TIFFTAG_XRESOLUTION, pixels_per_cm);
        set_field(TIFFTAG_YRESOLUTION, pixels_per_cm);

        // Set the image description
        std::string description = "{\"shape\": [" + std::to_string(height) + ", " + std::to_string(width) + ", " + std::to_string(channels) + "]}";
        set_field(TIFFTAG_IMAGEDESCRIPTION, description.c_str());

        // Set the software tag with version from dlup
        std::string software_tag = "dlup " + get_version() + " (libtiff " + std::to_string(TIFFLIB_VERSION) + ")";
        set_field(TIFFTAG_SOFTWARE, software_tag.c_str());
    }

    std::string get_version() const {
        py::module_ dlup = py::module_::import("dlup");
        return dlup.attr("__version__").cast<std::string>();
    }
};

PYBIND11_MODULE(fast_tiff_writer, m) {
    py::class_<FastTiffWriter>(m, "FastTiffWriter")
        .def(py::init([](py::object path, std::array<int, 3> size, double mpp,
                         std::array<int, 2> tile_size,
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

            return new FastTiffWriter(std::move(cpp_path), size, mpp, tile_size, comp_type, quality);
        }))
        .def("write_tile", &FastTiffWriter::write_tile);

    py::enum_<CompressionType>(m, "CompressionType")
        .value("NONE", CompressionType::NONE)
        .value("JPEG", CompressionType::JPEG)
        .value("LZW", CompressionType::LZW)
        .value("DEFLATE", CompressionType::DEFLATE);


    py::register_exception<TiffException>(m, "TiffException");
    py::register_exception<TiffOpenException>(m, "TiffOpenException");
    py::register_exception<TiffWriteException>(m, "TiffWriteException");
    py::register_exception<TiffSetupException>(m, "TiffSetupException");
}
