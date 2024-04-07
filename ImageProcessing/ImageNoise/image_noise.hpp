#pragma once

#include <Magick++.h>

namespace image_processing {
Magick::Image GenerateSaltNoise(const Magick::Image &image);
Magick::Image GenerateNormalNoise(const Magick::Image &image);
Magick::Image AntiNoiseMean(const Magick::Image &image, uint R);
Magick::Image AntiNoiseMedian(const Magick::Image &image, uint R);
} // namespace image_processing