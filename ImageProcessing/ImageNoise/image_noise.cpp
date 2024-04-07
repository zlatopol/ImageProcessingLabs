#include "image_noise.hpp"

#include "utility.hpp"

#include <random>

namespace image_processing {
Magick::Image GenerateSaltNoise(const Magick::Image &image) {
  std::default_random_engine e1(std::random_device{}());
  std::uniform_int_distribution<> uniform_dist(1, 99);

  return ProcessGrayPixel(
      [&](auto pixel) {
        auto rand_val = uniform_dist(e1);
        if (rand_val > 98) {
          return Magick::ColorGray{1. * ((rand_val - 90) / 5)};
        } else {
          return pixel;
        }
      },
      image);
}

Magick::Image GenerateNormalNoise(const Magick::Image &image) {
  std::default_random_engine e1(std::random_device{}());
  std::normal_distribution<> uniform_dist(-0.0002, 0.0002);

  return ProcessGrayPixel(
      [&](auto pixel) {
        auto rand_val = uniform_dist(e1);
        return Magick::ColorGray{
            std::max(0., std::min(1., pixel.shade() + rand_val))};
      },
      image);
}

Magick::Image AntiNoiseMean(const Magick::Image &image, uint R) {
  Magick::Image result = image;
  result.resize(result.size(), Magick::FilterTypes::BoxFilter, R);
  return result;
}

Magick::Image AntiNoiseMedian(const Magick::Image &image, uint R) {
  Magick::Image result = image;
  result.medianFilter(R);
  return result;
}

} // namespace image_processing
