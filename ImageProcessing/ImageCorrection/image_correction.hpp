#pragma once

#include <Magick++.h>

namespace image_processing {
Magick::Image Negative(const Magick::Image &image);
Magick::Image GammaCorrection(const Magick::Image &image, float C, float y);
Magick::Image LogCorrection(const Magick::Image &image, float C);
Magick::Image Equalize(const Magick::Image &image);
} // namespace image_processing
