#include "bilinear_interpolation.hpp"

#include <cassert>
#include <cmath>
#include <utility>

namespace {
float BilinearInterpolation(float q11, float q12, float q21, float q22,
                            float x1, float x2, float y1, float y2, float x,
                            float y) {
  float x2x1, y2y1, x2x, y2y, yy1, xx1;
  x2x1 = x2 - x1;
  y2y1 = y2 - y1;
  x2x = x2 - x;
  y2y = y2 - y;
  yy1 = y - y1;
  xx1 = x - x1;
  return 1.0 / (x2x1 * y2y1) *
         (q11 * x2x * y2y + q21 * xx1 * y2y + q12 * x2x * yy1 +
          q22 * xx1 * yy1);
}
} // namespace

namespace image_processing {
Magick::Image ResizeBilinear(const Magick::Image &image,
                             const Magick::Geometry &new_geometry) {
  const auto start_geometry = image.size();
  auto input_pixels = image.getConstPixels(0, 0, start_geometry.width(),
                                           start_geometry.height());

  Magick::Image result{new_geometry, Magick::ColorRGB{0, 0, 0}};

  result.modifyImage();
  result.type(Magick::GrayscaleType);
  auto output_pixels =
      result.getPixels(0, 0, new_geometry.width(), new_geometry.height());
  const auto get_off = [](const Magick::Geometry &geo, int x, int y) {
    return y * geo.width() + x;
  };

  for (std::size_t x = 0, y = 0; y < new_geometry.height(); x++) {
    if (x > new_geometry.width()) {
      x = 0;
      y++;
    }

    double gx =
        x / (double)(new_geometry.width()) * (start_geometry.width() - 1);
    double gy =
        y / (double)(new_geometry.height()) * (start_geometry.height() - 1);

    int gxi = (int)gx;
    int gyi = (int)gy;

    const auto pixel_c00 = input_pixels + get_off(start_geometry, gxi, gyi);
    const auto pixel_c10 = input_pixels + get_off(start_geometry, gxi + 1, gyi);
    const auto pixel_c01 = input_pixels + get_off(start_geometry, gxi, gyi + 1);
    const auto pixel_c11 =
        input_pixels + get_off(start_geometry, gxi + 1, gyi + 1);

    auto *pixel = output_pixels + get_off(new_geometry, x, y);
    auto bin_val = BilinearInterpolation(pixel_c00->blue, pixel_c01->blue,
                                         pixel_c10->blue, pixel_c11->blue, gxi,
                                         gxi + 1, gyi, gyi + 1, gx, gy);
    *pixel = Magick::ColorGray(bin_val / std::numeric_limits<Magick::Quantum>::max());
  }

  result.syncPixels();

  return result;
}
} // namespace image_processing
