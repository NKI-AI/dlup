#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

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

namespace fs = std::filesystem;
namespace py = pybind11;

class TiffException : public std::runtime_error {
public:
    explicit TiffException(const std::string &message) : std::runtime_error(message) {}
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

enum class CompressionType { NONE, JPEG, LZW, DEFLATE };

CompressionType string_to_compression_type(const std::string &compression) {
    if (compression == "NONE")
        return CompressionType::NONE;
    if (compression == "JPEG")
        return CompressionType::JPEG;
    if (compression == "LZW")
        return CompressionType::LZW;
    if (compression == "DEFLATE")
        return CompressionType::DEFLATE;
    throw std::invalid_argument("Invalid compression type: " + compression);
}

struct TIFFDeleter {
    void operator()(TIFF *tif) const noexcept {
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
    FastTiffWriter(fs::path filename, std::array<int, 3> size, double mpp, std::array<int, 2> tileSize = {512, 512},
                   CompressionType compression = CompressionType::JPEG, int quality = 100, bool pyramid = false)
        : filename(std::move(filename)), size(size), mpp(mpp), tileSize(tileSize), compression(compression),
          quality(quality), pyramid(pyramid), tif(nullptr) {

        validateInputs();

        TIFF *tiff_ptr = TIFFOpen(this->filename.c_str(), "w");
        if (!tiff_ptr) {
            throw TiffOpenException("Unable to create TIFF file");
        }
        tif.reset(tiff_ptr);

        setupTIFFDirectory(0);
    }

    ~FastTiffWriter();
    void writeTile(py::array_t<std::byte, py::array::c_style | py::array::forcecast> tile, int row, int col);
    void flush();
    void finalize();
    void writePyramid();

private:
    std::string filename;
    std::array<int, 3> size;
    double mpp;
    std::array<int, 2> tileSize;
    CompressionType compression;
    int quality;
    bool pyramid;
    int numLevels = calculateLevels();
    TIFFPtr tif;

    void validateInputs() const;
    int calculateLevels();
    void setupTIFFDirectory(int level);
    void writeTIFFDirectory();
    void writeDownsampledResolutionPage(int level);

    std::pair<uint32_t, uint32_t> getLevelDimensions(int level);
    std::string getDlupVersion() const;
};

FastTiffWriter::~FastTiffWriter() { finalize(); }

void FastTiffWriter::writeTile(py::array_t<std::byte, py::array::c_style | py::array::forcecast> tile, int row,
                               int col) {
    auto buf = tile.request();
    if (buf.ndim < 2 || buf.ndim > 3) {
        throw TiffWriteException("Invalid number of dimensions in tile data. Expected 2 or 3, got " +
                                 std::to_string(buf.ndim));
    }
    auto [height, width, channels] = std::tuple{buf.shape[0], buf.shape[1], buf.ndim > 2 ? buf.shape[2] : 1};

    // Verify dimensions and buffer size
    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (static_cast<size_t>(buf.size) != expected_size) {
        throw TiffWriteException("Buffer size does not match expected size. Expected " + std::to_string(expected_size) +
                                 ", got " + std::to_string(buf.size));
    }

    // Check if tile coordinates are within bounds
    if (row < 0 || row >= size[0] || col < 0 || col >= size[1]) {
        throw TiffWriteException("Tile coordinates out of bounds for row " + std::to_string(row) + ", col " +
                                 std::to_string(col) + ". Image size is " + std::to_string(size[0]) + "x" +
                                 std::to_string(size[1]));
    }

    // Write the tile
    if (TIFFWriteTile(tif.get(), buf.ptr, col, row, 0, 0) < 0) {
        throw TiffWriteException("TIFFWriteTile failed for row " + std::to_string(row) + ", col " +
                                 std::to_string(col));
    }
}

void FastTiffWriter::validateInputs() const {
    if (size[0] <= 0 || size[1] <= 0 || size[2] <= 0) {
        throw std::invalid_argument("Invalid size parameters");
    }
    if (mpp <= 0) {
        throw std::invalid_argument("Invalid mpp value");
    }
    if (tileSize[0] <= 0 || tileSize[1] <= 0) {
        throw std::invalid_argument("Invalid tile size");
    }
    if (quality < 0 || quality > 100) {
        throw std::invalid_argument("Invalid quality value");
    }
}

int FastTiffWriter::calculateLevels() {
    int maxDim = std::max(size[0], size[1]);
    int minTileDim = std::min(tileSize[0], tileSize[1]);
    int numLevels = 1;
    while (maxDim > minTileDim) {
        maxDim /= 2;
        numLevels++;
    }
    return numLevels;
}

std::string FastTiffWriter::getDlupVersion() const {
    py::module_ dlup = py::module_::import("dlup");
    return dlup.attr("__version__").cast<std::string>();
}

std::pair<uint32_t, uint32_t> FastTiffWriter::getLevelDimensions(int level) {
    uint32_t width = std::max(1, size[1] >> level);
    uint32_t height = std::max(1, size[0] >> level);
    return {width, height};
}

void FastTiffWriter::flush() {
    if (tif) {
        if (TIFFFlush(tif.get()) != 1) {
            throw TiffWriteException("Failed to flush TIFF file");
        }
    }
}

void FastTiffWriter::finalize() {
    if (tif) {
        // Only write directory if we haven't written all directories yet
        if (TIFFCurrentDirectory(tif.get()) < TIFFNumberOfDirectories(tif.get()) - 1) {
            TIFFWriteDirectory(tif.get());
        }
        TIFFClose(tif.get());
        tif.release();
    }
}

void FastTiffWriter::setupTIFFDirectory(int level) {
    auto set_field = [this](uint32_t tag, auto... value) {
        if (TIFFSetField(tif.get(), tag, value...) != 1) {
            throw TiffSetupException("Failed to set TIFF field: " + std::to_string(tag));
        }
    };

    auto [width, height] = getLevelDimensions(level);
    int channels = size[2];

    set_field(TIFFTAG_IMAGEWIDTH, width);
    set_field(TIFFTAG_IMAGELENGTH, height);
    set_field(TIFFTAG_SAMPLESPERPIXEL, channels);
    set_field(TIFFTAG_BITSPERSAMPLE, 8);
    set_field(TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    set_field(TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    set_field(TIFFTAG_TILEWIDTH, tileSize[1]);
    set_field(TIFFTAG_TILELENGTH, tileSize[0]);

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
        //                set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_YCBCR);
        //                set_field(TIFFTAG_YCBCRSUBSAMPLING, 2, 2);
        //                set_field(TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
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
    std::string description = "TODO"; // {\"shape\": [" + std::to_string(height) + ", " + std::to_string(width) + ", " +
                                      // std::to_string(channels) + "]}";
    set_field(TIFFTAG_IMAGEDESCRIPTION, description.c_str());

    // Set the software tag with version from dlup
    std::string software_tag = "dlup " + getDlupVersion() + " (libtiff " + std::to_string(TIFFLIB_VERSION) + ")";
    set_field(TIFFTAG_SOFTWARE, software_tag.c_str());

    // Set SubFileType for pyramid levels
    if (level == 0) {
        set_field(TIFFTAG_SUBFILETYPE, 0);
    } else {
        set_field(TIFFTAG_SUBFILETYPE, FILETYPE_REDUCEDIMAGE);
    }
}

void FastTiffWriter::writeDownsampledResolutionPage(int level) {
    if (level <= 0 || level >= numLevels) {
        throw std::invalid_argument("Invalid level for downsampled resolution page");
    }

    // Read the previous level image
    auto [prevWidth, prevHeight] = getLevelDimensions(level - 1);
    int channels = size[2];
    std::vector<std::byte> prevImage(prevWidth * prevHeight * channels);

    // Set the directory to the previous level
    if (!TIFFSetDirectory(tif.get(), level - 1)) {
        throw TiffReadException("Failed to set directory to level " + std::to_string(level - 1));
    }

    // Read the previous level image
    if (TIFFIsTiled(tif.get())) {
        uint32_t tileWidth, tileHeight;
        TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &tileHeight);

        std::vector<uint8_t> tileBuf(TIFFTileSize(tif.get()));

        for (uint32_t row = 0; row < prevHeight; row += tileHeight) {
            for (uint32_t col = 0; col < prevWidth; col += tileWidth) {
                if (TIFFReadTile(tif.get(), tileBuf.data(), col, row, 0, 0) < 0) {
                    throw TiffReadException("Failed to read tile at row " + std::to_string(row) + ", col " +
                                            std::to_string(col));
                }

                // Copy tile data to prevImage
                for (uint32_t y = 0; y < tileHeight && (row + y) < prevHeight; ++y) {
                    for (uint32_t x = 0; x < tileWidth && (col + x) < prevWidth; ++x) {
                        for (int c = 0; c < channels; ++c) {
                            prevImage[((row + y) * prevWidth + (col + x)) * channels + c] =
                                std::byte(tileBuf[(y * tileWidth + x) * channels + c]);
                        }
                    }
                }
            }
        }
    } else {
        throw TiffReadException("Only tiled TIFF images are supported for downsampling");
    }

    // Calculate dimensions for the current level
    auto [width, height] = getLevelDimensions(level);

    // Downsample the image
    std::vector<std::byte> downsampledImage(width * height * channels);
    stbir_resize_uint8(reinterpret_cast<const uint8_t *>(prevImage.data()), prevWidth, prevHeight, 0,
                       reinterpret_cast<uint8_t *>(downsampledImage.data()), width, height, 0, channels);

    // Create a new directory for the downsampled level
    if (!TIFFWriteDirectory(tif.get())) {
        throw TiffWriteException("Failed to create new directory for downsampled image");
    }

    // Setup the TIFF directory for the downsampled level
    setupTIFFDirectory(level);

    // Calculate the number of tiles needed to cover the downsampled image
    int numTilesX = (width + tileSize[1] - 1) / tileSize[1];
    int numTilesY = (height + tileSize[0] - 1) / tileSize[0];

    // Write downsampled tiles
    for (int row = 0; row < numTilesY; ++row) {
        for (int col = 0; col < numTilesX; ++col) {
            std::vector<std::byte> tile(tileSize[0] * tileSize[1] * channels);

            // Copy data from downsampled image to tile
            for (uint32_t y = 0; y < tileSize[0] && (row * tileSize[0] + y) < height; ++y) {
                for (uint32_t x = 0; x < tileSize[1] && (col * tileSize[1] + x) < width; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        tile[(y * tileSize[1] + x) * channels + c] =
                            downsampledImage[((row * tileSize[0] + y) * width + (col * tileSize[1] + x)) * channels +
                                             c];
                    }
                }
            }

            if (TIFFWriteTile(tif.get(), reinterpret_cast<uint8_t *>(tile.data()), col * tileSize[1], row * tileSize[0],
                              0, 0) < 0) {
                throw TiffWriteException("Failed to write downsampled tile at level " + std::to_string(level) +
                                         ", row " + std::to_string(row) + ", col " + std::to_string(col));
            }
        }
    }
}

void FastTiffWriter::writePyramid() {
    numLevels = calculateLevels();

    // The base level (level 0) is already written, so we start from level 1
    for (int level = 1; level < numLevels; ++level) {
        writeDownsampledResolutionPage(level);
        flush();
    }
}

PYBIND11_MODULE(fast_tiff_writer, m) {
    py::class_<FastTiffWriter>(m, "FastTiffWriter")
        .def(py::init([](py::object path, std::array<int, 3> size, double mpp, std::array<int, 2> tileSize,
                         py::object compression, int quality, bool pyramid) {
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

            return new FastTiffWriter(std::move(cpp_path), size, mpp, tileSize, comp_type, quality);
        }))
        .def("write_tile", &FastTiffWriter::writeTile)
        .def("finalize", &FastTiffWriter::finalize)
        .def("write_pyramid", &FastTiffWriter::writePyramid)
        .def("flush", &FastTiffWriter::flush);

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
}
