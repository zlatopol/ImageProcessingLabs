#include "image_correction.hpp"

#include <cmath>

#include "utility.hpp"

namespace image_processing {
Magick::Image Negative(const Magick::Image &image) {
  return ProcessGrayPixel(
      [](Magick::ColorGray gray_pixel) {
        return Magick::ColorGray{1 - gray_pixel.shade()};
      },
      image);
}

Magick::Image GammaCorrection(const Magick::Image &image, float C, float y) {
  return ProcessGrayPixel(
      [&](Magick::ColorGray gray_pixel) {
        return Magick::ColorGray{
            std::min(1., C * std::pow(gray_pixel.shade(), y))};
      },
      image);
}

Magick::Image LogCorrection(const Magick::Image &image, float C) {
  return ProcessGrayPixel(
      [&](Magick::ColorGray gray_pixel) {
        return Magick::ColorGray{
            std::min(1., C * std::log(gray_pixel.shade() + 1))};
      },
      image);
}

Magick::Image Equalize(const Magick::Image &image) {
  int hist[256]{};
  int max{};
  const auto geometry = image.size();
  ProcessGrayPixel(
      [&](auto gray_pixel) {
        int val = gray_pixel.shade() * (std::size(hist) - 1);
        max = std::max(max, val);
        ++hist[val];
        return gray_pixel;
      },
      image);

  double norm_hist[std::size(hist)]{};

  std::transform(std::begin(hist), std::end(hist), std::begin(norm_hist),
                 [&](auto hist_el) {
                   return (double)hist_el /
                          (geometry.width() * geometry.height());
                 });

  double cdf[std::size(norm_hist)]{};
  cdf[0] = norm_hist[0];
  for (std::size_t i = 1; i < std::size(norm_hist); ++i) {
    cdf[i] = cdf[i - 1] + norm_hist[i];
  }

  return ProcessGrayPixel(
      [&](auto gray_pixel) {
        int val = gray_pixel.shade() * (std::size(cdf) - 1);
        return Magick::ColorGray{std::min(1., cdf[val])};
      },
      image);
}

} // namespace image_processing