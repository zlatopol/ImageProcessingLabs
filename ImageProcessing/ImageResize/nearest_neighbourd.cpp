#include "nearest_neighbourd.hpp"

#include <iostream>

namespace image_processing {
Magick::Image ResizeNearestNeighbourd(const Magick::Image &image,
                                      const Magick::Geometry &new_geometry) {
  const auto start_geometry = image.size();
  Magick::Image result{new_geometry, Magick::ColorRGB{0, 0, 0}};
  auto input_pixels = image.getConstPixels(0, 0, start_geometry.width(),
                                           start_geometry.height());

  result.modifyImage();
  result.type(Magick::GrayscaleType);
  auto output_pixels =
      result.getPixels(0, 0, new_geometry.width(), new_geometry.height());

  const auto x_ratio =
      (start_geometry.width() << 16) / new_geometry.width() + 1;
  const auto y_ratio =
      (start_geometry.height() << 16) / new_geometry.height() + 1;

  const auto get_off = [](const Magick::Geometry &geo, int x, int y) {
    return y * geo.width() + x;
  };

  for (std::size_t y = 0; y < new_geometry.height(); ++y) {
    for (std::size_t x = 0; x < new_geometry.width(); ++x) {
      const auto x2 = ((x * x_ratio) >> 16);
      const auto y2 = ((y * y_ratio) >> 16);

      auto *output_pixel = output_pixels + get_off(new_geometry, x, y);
      const auto *input_pixel = input_pixels + get_off(start_geometry, x2, y2);

      *output_pixel = *input_pixel;
    }
  }

  result.syncPixels();

  return result;
}
} // namespace image_processing
