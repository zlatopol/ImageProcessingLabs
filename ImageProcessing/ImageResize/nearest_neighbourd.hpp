#pragma once

#include <Magick++.h>

namespace image_processing {
Magick::Image ResizeNearestNeighbourd(const Magick::Image &image,
                                      const Magick::Geometry &new_geometry);
}
