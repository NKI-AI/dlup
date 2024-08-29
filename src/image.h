// image.h

#ifndef IMAGE_H
#define IMAGE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace image_utils {

void downsample2x2(const std::vector<std::byte> &input, uint32_t inputWidth, uint32_t inputHeight,
                   std::vector<std::byte> &output, uint32_t outputWidth, uint32_t outputHeight, int channels) {
    for (uint32_t y = 0; y < outputHeight; ++y) {
        for (uint32_t x = 0; x < outputWidth; ++x) {
            for (int c = 0; c < channels; ++c) {
                uint32_t sum = 0;
                uint32_t count = 0;
                for (uint32_t dy = 0; dy < 2; ++dy) {
                    for (uint32_t dx = 0; dx < 2; ++dx) {
                        uint32_t sx = 2 * x + dx;
                        uint32_t sy = 2 * y + dy;
                        if (sx < inputWidth && sy < inputHeight) {
                            sum += std::to_integer<uint32_t>(input[(sy * inputWidth + sx) * channels + c]);
                            ++count;
                        }
                    }
                }
                output[(y * outputWidth + x) * channels + c] = static_cast<std::byte>(sum / count);
            }
        }
    }
}

} // namespace image_utils

#endif // IMAGE_H
