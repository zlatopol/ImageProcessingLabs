#include "image_processing.hpp"

#include <cmath>
#include <limits>
#include <random>

#include "utility.hpp"

namespace image_processing {
Magick::Image Mult(const Magick::Image &image, float C) {
  return ProcessGrayPixel(
      [&](auto gray_pixel) {
        return Magick::ColorGray{std::min(1., gray_pixel.shade() * C)};
      },
      image);
}

Magick::Image AddConstant(const Magick::Image &image, uint8_t C) {
  return ProcessGrayPixel(
      [&](auto gray_pixel) {
        return Magick::ColorGray{std::min(
            1., gray_pixel.shade() +
                    (float)C / std::numeric_limits<decltype(C)>::max())};
      },
      image);
}

Magick::Image Diff(const Magick::Image &lhs, const Magick::Image &rhs) {
  return ProcessGrayPixel(
      [](auto lhs, auto rhs) {
        return Magick::ColorGray{std::abs(lhs.shade() - rhs.shade())};
      },
      lhs, rhs);
}
} // namespace image_processing