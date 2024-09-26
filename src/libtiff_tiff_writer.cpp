#include "constants.h"
#include "image.h"
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

#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

namespace fs = std::filesystem;
namespace py = pybind11;

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

enum class CompressionType { NONE, JPEG, LZW, DEFLATE, ZSTD };

CompressionType string_to_compression_type(const std::string &compression) {
    if (compression == "NONE")
        return CompressionType::NONE;
    if (compression == "JPEG")
        return CompressionType::JPEG;
    if (compression == "LZW")
        return CompressionType::LZW;
    if (compression == "DEFLATE")
        return CompressionType::DEFLATE;
    if (compression == "ZSTD")
        return CompressionType::ZSTD;
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

class LibtiffTiffWriter {
public:
    LibtiffTiffWriter(fs::path filename, std::array<int, 3> imageSize, std::array<float, 2> mpp,
                      std::array<int, 2> tileSize, CompressionType compression = CompressionType::JPEG,
                      int quality = 100)
        : filename(std::move(filename)), imageSize(imageSize), mpp(mpp), tileSize(tileSize), compression(compression),
          quality(quality), tif(nullptr) {

        validateInputs();

        TIFF *tiff_ptr = TIFFOpen(this->filename.c_str(), "w");
        if (!tiff_ptr) {
            throw TiffOpenException("Unable to create TIFF file");
        }
        tif.reset(tiff_ptr);

        setupTIFFDirectory(0);
    }

    ~LibtiffTiffWriter();
    void writeTile(py::array_t<std::byte, py::array::c_style | py::array::forcecast> tile, int row, int col);
    void flush();
    void finalize();
    void writePyramid();

private:
    std::string filename;
    std::array<int, 3> imageSize;
    std::array<float, 2> mpp;
    std::array<int, 2> tileSize;
    CompressionType compression;
    int quality;
    uint32_t tileCounter;
    int numLevels = calculateLevels();
    TIFFPtr tif;

    void validateInputs() const;
    int calculateLevels();
    std::pair<uint32_t, uint32_t> calculateTiles(int level);
    uint32_t calculateNumTiles(int level);
    void setupTIFFDirectory(int level);
    void writeTIFFDirectory();
    void writeDownsampledResolutionPage(int level);

    std::pair<uint32_t, uint32_t> getLevelDimensions(int level);
    std::vector<std::byte> read2x2TileGroup(TIFF *readTif, uint32_t row, uint32_t col, uint32_t prevWidth,
                                            uint32_t prevHeight);
    void setupReadTIFF(TIFF *readTif);
};

LibtiffTiffWriter::~LibtiffTiffWriter() { finalize(); }

void LibtiffTiffWriter::writeTile(py::array_t<std::byte, py::array::c_style | py::array::forcecast> tile, int row,
                                  int col) {
    auto numTiles = calculateNumTiles(0);
    if (tileCounter >= numTiles) {
        throw TiffWriteException("all tiles have already been written");
    }
    auto buf = tile.request();
    if (buf.ndim < 2 || buf.ndim > 3) {
        throw TiffWriteException("invalid number of dimensions in tile data. Expected 2 or 3, got " +
                                 std::to_string(buf.ndim));
    }
    auto [height, width, channels] = std::tuple{buf.shape[0], buf.shape[1], buf.ndim > 2 ? buf.shape[2] : 1};

    // Verify dimensions and buffer size
    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (static_cast<size_t>(buf.size) != expected_size) {
        throw TiffWriteException("buffer size does not match expected size. Expected " + std::to_string(expected_size) +
                                 ", got " + std::to_string(buf.size));
    }

    // Check if tile coordinates are within bounds
    if (row < 0 || row >= imageSize[0] || col < 0 || col >= imageSize[1]) {
        auto [imageWidth, imageHeight] = getLevelDimensions(0);
        throw TiffWriteException("tile coordinates out of bounds for row " + std::to_string(row) + ", col " +
                                 std::to_string(col) + ". Image size is " + std::to_string(imageWidth) + "x" +
                                 std::to_string(imageHeight));
    }

    // Write the tile
    if (TIFFWriteTile(tif.get(), buf.ptr, col, row, 0, 0) < 0) {
        throw TiffWriteException("TIFFWriteTile failed for row " + std::to_string(row) + ", col " +
                                 std::to_string(col));
    }
    tileCounter++;
    if (tileCounter == numTiles) {
        flush();
    }
}

void LibtiffTiffWriter::validateInputs() const {
    // check positivity of image size
    if (imageSize[0] <= 0 || imageSize[1] <= 0 || imageSize[2] <= 0) {
        throw std::invalid_argument("Invalid size parameters");
    }

    // check positivity of mpp
    if (mpp[0] <= 0 || mpp[1] <= 0) {
        throw std::invalid_argument("Invalid mpp value");
    }

    // check positivity of tile size
    if (tileSize[0] <= 0 || tileSize[1] <= 0) {
        throw std::invalid_argument("Invalid tile size");
    }

    // check quality parameter
    if (quality < 0 || quality > 100) {
        throw std::invalid_argument("Invalid quality value");
    }

    // check if tile size is power of two
    if ((tileSize[0] & (tileSize[0] - 1)) != 0 || (tileSize[1] & (tileSize[1] - 1)) != 0) {
        throw std::invalid_argument("Tile size must be a power of two");
    }
}

int LibtiffTiffWriter::calculateLevels() {
    int maxDim = std::max(imageSize[0], imageSize[1]);
    int minTileDim = std::min(tileSize[0], tileSize[1]);
    int numLevels = 1;
    while (maxDim > minTileDim * 2) {
        maxDim /= 2;
        numLevels++;
    }
    return numLevels;
}

std::pair<uint32_t, uint32_t> LibtiffTiffWriter::calculateTiles(int level) {
    auto [currentWidth, currentHeight] = getLevelDimensions(level);
    auto [tileWidth, tileHeight] = tileSize;

    uint32_t numTilesX = (currentWidth + tileWidth - 1) / tileWidth;
    uint32_t numTilesY = (currentHeight + tileHeight - 1) / tileHeight;
    return {numTilesX, numTilesY};
}

uint32_t LibtiffTiffWriter::calculateNumTiles(int level) {
    auto [numTilesX, numTilesY] = calculateTiles(level);
    return numTilesX * numTilesY;
}

std::pair<uint32_t, uint32_t> LibtiffTiffWriter::getLevelDimensions(int level) {
    uint32_t levelWidth = std::max(1, imageSize[1] >> level);
    uint32_t levelHeight = std::max(1, imageSize[0] >> level);
    return {levelWidth, levelHeight};
}

void LibtiffTiffWriter::flush() {
    if (tif) {
        if (TIFFFlush(tif.get()) != 1) {
            throw TiffWriteException("failed to flush TIFF file");
        }
    }
}

void LibtiffTiffWriter::finalize() {
    if (tif) {
        // Only write directory if we haven't written all directories yet
        if (TIFFCurrentDirectory(tif.get()) < TIFFNumberOfDirectories(tif.get()) - 1) {
            TIFFWriteDirectory(tif.get());
        }
        TIFFClose(tif.get());
        tif.release();
    }
}

void LibtiffTiffWriter::setupReadTIFF(TIFF *readTif) {
    auto set_field = [readTif](uint32_t tag, auto... value) {
        if (TIFFSetField(readTif, tag, value...) != 1) {
            throw TiffSetupException("failed to set TIFF field for reading: " + std::to_string(tag));
        }
    };

    uint16_t compression;
    if (TIFFGetField(readTif, TIFFTAG_COMPRESSION, &compression) == 1) {
        if (compression == COMPRESSION_JPEG) {
            set_field(TIFFTAG_JPEGCOLORMODE, JPEGCOLORMODE_RGB);
        }
    }
}

void LibtiffTiffWriter::setupTIFFDirectory(int level) {
    auto set_field = [this](uint32_t tag, auto... value) {
        if (TIFFSetField(tif.get(), tag, value...) != 1) {
            throw TiffSetupException("failed to set TIFF field: " + std::to_string(tag));
        }
    };

    auto [width, height] = getLevelDimensions(level);
    int channels = imageSize[2];

    set_field(TIFFTAG_IMAGEWIDTH, width);
    set_field(TIFFTAG_IMAGELENGTH, height);
    set_field(TIFFTAG_SAMPLESPERPIXEL, channels);
    set_field(TIFFTAG_BITSPERSAMPLE, 8);
    set_field(TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    set_field(TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    set_field(TIFFTAG_TILEWIDTH, tileSize[1]);
    set_field(TIFFTAG_TILELENGTH, tileSize[0]);

    if (channels == 3 || channels == 4) {
        if (compression != CompressionType::JPEG) {
            set_field(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        }
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
    case CompressionType::ZSTD:
#ifdef HAVE_ZSTD
        set_field(TIFFTAG_COMPRESSION, COMPRESSION_ZSTD);
        set_field(TIFFTAG_ZSTD_LEVEL, 3); // 3 is the default
        break;
#else
        throw TiffCompressionNotSupportedError("ZSTD");
#endif
    default:
        throw TiffSetupException("Unknown compression type");
    }

    // Convert mpp (micrometers per pixel) to pixels per centimeter
    double pixels_per_cm_x = 10000.0 / mpp[0];
    double pixels_per_cm_y = 10000.0 / mpp[1];

    set_field(TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
    set_field(TIFFTAG_XRESOLUTION, pixels_per_cm_x);
    set_field(TIFFTAG_YRESOLUTION, pixels_per_cm_y);

    // Set the image description
    // TODO: This needs to be configurable
    std::string description = "TODO";
    //    set_field(TIFFTAG_IMAGEDESCRIPTION, description.c_str());

    // Set the software tag with version from dlup
    std::string software_tag =
        "dlup " + std::string(DLUP_VERSION) + " (libtiff " + std::to_string(TIFFLIB_VERSION) + ")";
    set_field(TIFFTAG_SOFTWARE, software_tag.c_str());

    // Set SubFileType for pyramid levels
    if (level == 0) {
        set_field(TIFFTAG_SUBFILETYPE, 0);
    } else {
        set_field(TIFFTAG_SUBFILETYPE, FILETYPE_REDUCEDIMAGE);
    }
}

std::vector<std::byte> LibtiffTiffWriter::read2x2TileGroup(TIFF *readTif, uint32_t row, uint32_t col,
                                                           uint32_t prevWidth, uint32_t prevHeight) {
    auto [tileWidth, tileHeight] = tileSize;
    int channels = imageSize[2];
    uint32_t fullGroupWidth = 2 * tileWidth;
    uint32_t fullGroupHeight = 2 * tileHeight;

    // Initialize a zero buffer for the 2x2 group
    std::vector<std::byte> groupBuffer(fullGroupWidth * fullGroupHeight * channels, std::byte(0));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            uint32_t tileRow = row + i * tileHeight;
            uint32_t tileCol = col + j * tileWidth;

            // Skip if this tile is out of bounds, this can happen when the image dimensions are smaller than 2x2 in
            // tileSize
            if (tileRow >= prevHeight || tileCol >= prevWidth) {
                continue;
            }

            std::vector<uint8_t> tileBuf(TIFFTileSize(readTif));

            if (TIFFReadTile(readTif, tileBuf.data(), tileCol, tileRow, 0, 0) < 0) {
                throw TiffReadException("failed to read tile at row " + std::to_string(tileRow) + ", col " +
                                        std::to_string(tileCol));
            }

            // Copy tile data to groupBuffer
            uint32_t copyWidth = std::min<uint32_t>(tileWidth, prevWidth - tileCol);
            uint32_t copyHeight = std::min<uint32_t>(tileHeight, prevHeight - tileRow);
            for (uint32_t y = 0; y < copyHeight; ++y) {
                for (uint32_t x = 0; x < copyWidth; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        size_t groupIndex =
                            ((i * tileHeight + y) * fullGroupWidth + (j * tileWidth + x)) * channels + c;
                        size_t tileIndex = (y * tileWidth + x) * channels + c;
                        groupBuffer[groupIndex] = static_cast<std::byte>(tileBuf[tileIndex]);
                    }
                }
            }
        }
    }

    return groupBuffer;
}
void LibtiffTiffWriter::writeDownsampledResolutionPage(int level) {
    if (level <= 0 || level >= numLevels) {
        throw std::invalid_argument("Invalid level for downsampled resolution page");
    }

    auto [prevWidth, prevHeight] = getLevelDimensions(level - 1);
    int channels = imageSize[2];
    auto [tileWidth, tileHeight] = tileSize;

    TIFFPtr readTif(TIFFOpen(filename.c_str(), "r"));
    if (!readTif) {
        throw TiffOpenException("failed to open TIFF file for reading");
    }

    if (!TIFFSetDirectory(readTif.get(), level - 1)) {
        throw TiffReadException("failed to set directory to level " + std::to_string(level - 1));
    }
    setupReadTIFF(readTif.get());

    if (!TIFFSetDirectory(tif.get(), level - 1)) {
        throw TiffReadException("failed to set directory to level " + std::to_string(level - 1));
    }

    if (!TIFFWriteDirectory(tif.get())) {
        throw TiffWriteException("failed to create new directory for downsampled image");
    }

    setupTIFFDirectory(level);

    auto [numTilesX, numTilesY] = calculateTiles(level);

    for (uint32_t tileY = 0; tileY < numTilesY; ++tileY) {
        for (uint32_t tileX = 0; tileX < numTilesX; ++tileX) {
            uint32_t row = tileY * tileHeight * 2;
            uint32_t col = tileX * tileWidth * 2;

            std::vector<std::byte> groupBuffer = read2x2TileGroup(readTif.get(), row, col, prevWidth, prevHeight);
            std::vector<std::byte> downsampledBuffer(tileHeight * tileWidth * channels);

            image_utils::downsample2x2(groupBuffer, 2 * tileWidth, 2 * tileHeight, downsampledBuffer, tileWidth,
                                       tileHeight, channels);

            if (TIFFWriteTile(tif.get(), reinterpret_cast<uint8_t *>(downsampledBuffer.data()), tileX * tileWidth,
                              tileY * tileHeight, 0, 0) < 0) {
                throw TiffWriteException("failed to write downsampled tile at level " + std::to_string(level) +
                                         ", row " + std::to_string(tileY) + ", col " + std::to_string(tileX));
            }
        }
    }

    readTif.reset();
    flush();
}

void LibtiffTiffWriter::writePyramid() {
    numLevels = calculateLevels();

    // The base level (level 0) is already written, so we start from level 1
    for (int level = 1; level < numLevels; ++level) {
        writeDownsampledResolutionPage(level);
        flush();
    }
}

PYBIND11_MODULE(_libtiff_tiff_writer, m) {
    py::class_<LibtiffTiffWriter>(m, "LibtiffTiffWriter")
        .def(py::init([](py::object path, std::array<int, 3> size, std::array<float, 2> mpp,
                         std::array<int, 2> tileSize, py::object compression, int quality) {
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

            return new LibtiffTiffWriter(std::move(cpp_path), size, mpp, tileSize, comp_type, quality);
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
