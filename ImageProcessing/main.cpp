#define FMT_HEADER_ONLY

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <utility>

#include <Magick++.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <matplot/matplot.h>

#include "ImageCorrection/image_correction.hpp"
#include "ImageNoise/image_noise.hpp"
#include "ImageProcessing/image_processing.hpp"
#include "ImageResize/bilinear_interpolation.hpp"
#include "ImageResize/nearest_neighbourd.hpp"
#include "utility.hpp"

void PrintImageGeometry(const Magick::Image &image) {
  Magick::Geometry geometry = image.size();

  std::cout << fmt::format("width: {}\nheight: {}", geometry.width(),
                           geometry.height())
            << std::endl;
}

Magick::Image readXCRFile(const std::filesystem::path &path,
                          const Magick::Geometry &geometry) {
  static const int PIXEL_SIZE = 2;
  Magick::Image image;
  image.size(geometry);
  image.magick("GRAY");
  image.depth(8);

  std::ifstream input_file(path, std::ios::binary);

  input_file.seekg(2048);

  const auto data_size = geometry.height() * geometry.width();

  auto data = std::make_unique<unsigned short[]>(data_size);

  input_file.read((char *)data.get(), data_size * PIXEL_SIZE);

  for (std::size_t i = 0; i < data_size; ++i) {
    auto *bytes = (unsigned char *)&data[i];
    std::swap(bytes[0], bytes[1]);
  }

  auto recalced_data = std::make_unique<unsigned char[]>(data_size);
  const auto min = *std::min_element(data.get(), data.get() + data_size);
  const auto max = *std::max_element(data.get(), data.get() + data_size);

  for (std::size_t i = 0; i < data_size; ++i) {
    recalced_data[i] =
        (unsigned char)(1.f * (data[i] - min) / (max - min) *
                        std::numeric_limits<unsigned char>::max());
  }

  Magick::Blob blob;
  blob.updateNoCopy(recalced_data.release(), data_size);

  image.read(blob);

  return image;
}

Magick::Image readFile(const std::filesystem::path &path,
                       std::optional<Magick::Geometry> geometry) {
  if (path.extension() == ".xcr") {
    if (geometry) {
      return readXCRFile(path, *geometry);
    } else {
      throw std::logic_error{"geometry must be specified for .xcr files"};
    }
  } else if (path.extension() == ".dat" || path.extension() == ".bin") {
    if (geometry) {
      if (path.extension() == ".dat") {
        return image_processing::VecToImage(
            image_processing::FormImage(image_processing::ReadDataFile(path),
                                        geometry->height(), geometry->width()));
      } else {
        return image_processing::VecToImage(image_processing::FormImage(
            image_processing::ReadDataFile<uint16_t>(path), geometry->height(),
            geometry->width()));
      }
    } else {
      throw std::logic_error{"geometry must be specified for .xcr files"};
    }
  } else {
    return Magick::Image{path};
  }
}

template <class T> Magick::Geometry geometryFromVec(const T &vec) {
  return {vec[0], vec[1]};
}

int main(int argc, char **argv) {
  Magick::InitializeMagick(nullptr);

  argparse::ArgumentParser program("ImageProcessing");

  program.add_argument("image_path").help("path to the image to work with");
  program.add_argument("--ai").help(
      "specify additional image_path for commands");

  program.add_argument("-t", "--type")
      .required()
      .help(
          "specify the programm's behaviour: geo - prints image's geometry; "
          "shift - add constant to all pixels in image; "
          "mult - multiply by constant all pixels in image; "
          "rotate - rotate images on 90 degree; "
          "recalc - recalculate image pixels to range [0;255]; "
          "resize - resize image; "
          "gamma - apply gamma correction; "
          "equalize - equalize image; "
          "diff - calc difference between images; "
          "salt_noise - add salt noise to image; "
          "norm_noise - add normed noise to image; "
          "comb_noise - add salt + normed noises to image; "
          "mean_antinoise - filter out noise from image with mean filter; "
          "median_antinoise - filter out noise from image with median filter; "
          "lab81; lab82; lab82_inverse; "
          "lab91; lab92");

  program.add_argument("-s", "--size")
      .help("specify width and height: --size width height. Required *.xcr "
            "images")
      .nargs(2)
      .scan<'u', unsigned int>();

  program.add_argument("-ns", "--new-size")
      .help("specify new size for image (required for resize type)")
      .nargs(2)
      .scan<'u', unsigned int>();

  program.add_argument("-rt", "--resize-type")
      .default_value("nearest")
      .help("specify type of resizing: nearest or bilinear. Default: nearest");

  program.add_argument("-gt", "--gamma-type")
      .default_value("negative")
      .help("specify function for gamma correction: negative (default), gamma, "
            "logarithm");

  program.add_argument("-C")
      .help("specify parameter for gamma correction, multipling and shifting")
      .scan<'g', double>()
      .default_value(1.);

  program.add_argument("-y")
      .help("specify parameter for gamma correction")
      .scan<'g', double>()
      .default_value(2.2);

  program.add_argument("-R")
      .help("specify parameter for noise filtering")
      .scan<'u', unsigned int>()
      .default_value(1.);

  program.add_argument("-o", "--output")
      .help("must be specified with image modifing functions");

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const auto size_vec = program.present<std::vector<unsigned int>>("--size");
  const auto geometry = size_vec
                            ? std::optional{geometryFromVec(size_vec.value())}
                            : std::nullopt;
  const auto image_path = program.get<std::string>("image_path");
  const auto additional_ip = [&]() -> std::optional<std::string> {
    if (program.is_used("--ai")) {
      return program.get<std::string>("--ai");
    } else {
      return std::nullopt;
    }
  }();
  const auto runtime_type = program.get<std::string>("--type");
  auto image = readFile(image_path, geometry);
  const auto additional_image = [&]() -> std::optional<Magick::Image> {
    if (additional_ip) {
      return readFile(*additional_ip, geometry);
    } else {
      return std::nullopt;
    }
  }();

  if (runtime_type == "geo") {
    PrintImageGeometry(image);
  } else if (runtime_type == "shift") {
    image = image_processing::AddConstant(image,
                                          (uint8_t)program.get<double>("-C"));
  } else if (runtime_type == "mult") {
    image = image_processing::Mult(image, program.get<double>("-C"));
  } else if (runtime_type == "rotate") {
    image.rotate(270);
  } else if (runtime_type == "recalc") {
  } else if (runtime_type == "resize") {
    if (program.is_used("-ns")) {
      const auto res_type = program.get<std::string>("--resize-type");
      if (res_type == "nearest") {
        image = image_processing::ResizeNearestNeighbourd(
            image,
            geometryFromVec(program.get<std::vector<unsigned int>>("-ns")));
      } else if (res_type == "bilinear") {
        image = image_processing::ResizeBilinear(
            image,
            geometryFromVec(program.get<std::vector<unsigned int>>("-ns")));
      } else {
        std::cerr << "Unknown resize type" << std::endl;
        return 1;
      }
    } else {
      std::cerr << "--new-size: required for resize action" << std::endl;
      return 1;
    }
  } else if (runtime_type == "gamma") {
    const auto gamma_type = program.get<std::string>("--gamma-type");
    const auto c_param = program.get<double>("-C");
    const auto y_param = program.get<double>("-y");
    if (gamma_type == "negative") {
      image = image_processing::Negative(image);
    } else if (gamma_type == "gamma") {
      image = image_processing::GammaCorrection(image, c_param, y_param);
    } else if (gamma_type == "logarithm") {
      image = image_processing::LogCorrection(image, c_param);
    } else {
      std::cerr << "Unhandled --gt: " << runtime_type << std::endl;
      return 1;
    }
  } else if (runtime_type == "equalize") {
    image = image_processing::Equalize(image);
  } else if (runtime_type == "diff") {
    if (!additional_image) {
      std::cerr << "--ai: required for diff action" << std::endl;
      return 1;
    }
    image = image_processing::Diff(image, *additional_image);
  } else if (runtime_type == "salt_noise") {
    image = image_processing::GenerateSaltNoise(image);
  } else if (runtime_type == "norm_noise") {
    image = image_processing::GenerateNormalNoise(image);
  } else if (runtime_type == "comb_noise") {
    image = image_processing::GenerateNormalNoise(
        image_processing::GenerateSaltNoise(image));
  } else if (runtime_type == "mean_antinoise") {
    auto R = program.get<unsigned int>("-R");
    image = image_processing::AntiNoiseMean(image, R);
  } else if (runtime_type == "median_antinoise") {
    auto R = program.get<unsigned int>("-R");
    image = image_processing::AntiNoiseMedian(image, R);
  } else if (runtime_type == "lab81") {
    double A = 100;
    double f = 50;
    size_t N = 1024;
    double dt = 0.001;

    std::vector<double> x1 = matplot::linspace(0, 2 * matplot::pi, N);
    auto y1 = image_processing::HarmFunction(A, f, N, dt);
    auto [x2, y2] =
        image_processing::SpecterFourier(image_processing::Fourier(y1), dt);
    auto inverse_harm = image_processing::InverseFourier(
        image_processing::FourierWithoutSquare(y1));
    auto [x3, y3] = image_processing::SpecterFourier(
        image_processing::Fourier(inverse_harm), dt);

    matplot::tiledlayout(2, 2);
    auto ax1 = matplot::nexttile();
    matplot::plot(ax1, x1, y1);
    matplot::title(ax1, "Harmonic func");

    auto ax2 = matplot::nexttile();
    matplot::plot(ax2, x2, y2);
    matplot::title(ax2, "Fourier harm");

    auto ax3 = matplot::nexttile();
    matplot::plot(ax3, x1, inverse_harm);
    matplot::title(ax3, "Invers Fourier harm");

    auto ax4 = matplot::nexttile();
    matplot::plot(ax4, x3, y3);
    matplot::title(ax4, "Fourier specter inverse harm");

    matplot::show();
  } else if (runtime_type == "lab82") {
    auto vec =
        image_processing::Fourier2D(image_processing::ImageTo2dVec(image));
    vec = image_processing::DiagonalShift2d(vec, vec.size() / 2,
                                            vec[0].size() / 2);
    image = image_processing::VecToImage(vec);
    image = image_processing::LogCorrection(image, 6);
  } else if (runtime_type == "lab82_inverse") {
    auto vec = image_processing::Fourier2DWithoutSquare(
        image_processing::ImageTo2dVec(image));
    image =
        image_processing::VecToImage(image_processing::InverseFourier2D(vec));
  } else if (runtime_type == "lab91") {
    auto [_, h] = image_processing::CreateHeartBeat();
    std::vector<double> x_c(1000);
    x_c[200] = 1 + 0.1;
    x_c[400] = 1 - 0.1;
    x_c[600] = 1 + 0.1;
    x_c[800] = 1 - 0.1;

    auto x = matplot::linspace(0, 1000, 1000);

    auto y = image_processing::Convolution(x_c, h);
    y.resize(1000);

    matplot::tiledlayout(2, 2);
    auto ax1 = matplot::nexttile();
    matplot::plot(ax1, x, y);
    matplot::title(ax1, "Normal Heartbeat");

    auto ax2 = matplot::nexttile();
    matplot::plot(ax2, x, x_c);

    auto ax3 = matplot::nexttile();
    auto y2 = image_processing::InverseFilter(y, h);
    matplot::plot(ax3, matplot::linspace(0, y2.size(), y2.size()), y2);
    matplot::title(ax3, "Inverse Normal Heartbeat");

    matplot::show();

    auto noised = y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1., 1.);

    for (auto &el : noised) {
      el += dis(gen);
    }

    matplot::tiledlayout(1, 2);
    auto ax4 = matplot::nexttile();
    matplot::plot(ax4, x, noised);
    matplot::title(ax4, "Noised Heartbeat");

    auto ax5 = matplot::nexttile();
    auto y3 = image_processing::InverseFilter(noised, h);
    matplot::plot(ax5, matplot::linspace(0, y3.size(), y3.size()), y3);
    matplot::title(ax5, "Inverse Noised Heartbeat");

    matplot::show();
  } else if (runtime_type == "lab92") {
    auto h_func = image_processing::ReadDataFile(
        "/Users/v.shlyaga/ImageProcessingLabs/images/kern76D.dat");
    h_func.resize(75);

    matplot::plot(h_func);
    matplot::show();
  } else if (runtime_type == "lab92_inv") {
    auto h_func = image_processing::ReadDataFile(
        "/Users/v.shlyaga/ImageProcessingLabs/images/kern76D.dat");

    auto img_data = image_processing::ImageTo2dVec(image);
    image = image_processing::VecToImage(
        image_processing::ImageRestore(img_data, h_func));
  } else if (runtime_type == "lab92_inv_noised") {
    auto h_func = image_processing::ReadDataFile(
        "/Users/v.shlyaga/ImageProcessingLabs/images/kern76D.dat");

    auto img_data = image_processing::ImageTo2dVec(image);
    image = image_processing::VecToImage(
        image_processing::ImageRestore(img_data, h_func, true, 0.1));
  } else if (runtime_type == "lab10_1") {
    auto vec = image_processing::Fourier2DWithoutSquare(
        image_processing::ImageTo2dVec(image), false);

    const auto c_param = program.get<double>("-C");

    vec = image_processing::Upsize2dVec(vec, c_param);

    image =
        image_processing::VecToImage(image_processing::InverseFourier2D(vec));
  } else if (runtime_type == "lab10_2") {
    auto vec = image_processing::ImageTo2dVec(image);

    const auto c_param = program.get<double>("-C");

    const auto fc = c_param * 0.5;
    const std::size_t m = 64;

    // auto lpf_data = image_processing::LpfData(fc, m);
    // vec = image_processing::PassFilter2d(vec, lpf_data, m);

    vec = image_processing::Fourier2DWithoutSquare(vec);

    vec = image_processing::Downsize2dVec(vec, c_param);

    image =
        image_processing::VecToImage(image_processing::InverseFourier2D(vec));
  } else if (runtime_type == "lab12_1p") {
    std::vector<std::vector<double>> kernel_x = {
        {-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

    std::vector<std::vector<double>> kernel_y = {
        {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};

    auto vec = image_processing::ImageTo2dVec(image);

    vec = image_processing::Convolution2D(vec, kernel_x, kernel_y);
    image_processing::ThresholdOp(vec);

    image = image_processing::VecToImage(vec);
  } else if (runtime_type == "lab12_1s") {
    std::vector<std::vector<double>> kernel_x = {
        {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    std::vector<std::vector<double>> kernel_y = {
        {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    auto vec = image_processing::ImageTo2dVec(image);

    vec = image_processing::Convolution2D(vec, kernel_x, kernel_y);
    image_processing::ThresholdOp(vec);

    image = image_processing::VecToImage(vec);
  } else if (runtime_type == "lab12_1l") {
    std::vector<std::vector<double>> kernel_x = {
        {1, 1, 1}, {1, -8, 1}, {1, 1, 1}};

    // std::vector<std::vector<double>> kernel_y = {
    //     {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};
    std::vector<std::vector<double>> kernel_y(3, std::vector<double>(3));

    auto vec = image_processing::ImageTo2dVec(image);

    vec = image_processing::Convolution2D(vec, kernel_x, kernel_y);
    image_processing::ThresholdOp(vec);

    image = image_processing::VecToImage(vec);
  } else if (runtime_type == "lab12_2") {
    std::vector<std::vector<double>> kernel_x = {
        {1, 1, 1}, {1, -8, 1}, {1, 1, 1}};

    std::vector<std::vector<double>> kernel_y = {
        {-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

    auto vec = image_processing::ImageTo2dVec(image);
    const auto cp = vec;

    vec = image_processing::Convolution2D(vec, kernel_x, kernel_y);

    image = image_processing::VecToImage(image_processing::DiffModel(cp, vec));
  } else if (runtime_type == "lab13_eros") {
    auto vec = image_processing::ImageTo2dVec(image);

    const auto r_param = program.get<double>("-R");

    image_processing::ThresholdOp(vec);

    auto cp = vec;

    vec = image_processing::Erode(vec, r_param);

    image = image_processing::VecToImage(image_processing::DiffModel(cp, vec));
  } else if (runtime_type == "lab13_dilate") {
    auto vec = image_processing::ImageTo2dVec(image);

    image_processing::ThresholdOp(vec);

    auto cp = vec;

    vec = image_processing::Dilate(vec, 3);

    image = image_processing::VecToImage(image_processing::DiffModel(cp, vec));
  } else if (runtime_type == "spec_1") {
    // std::vector<std::vector<double>> kernel_x = {
    //     {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // std::vector<std::vector<double>> kernel_y = {
    //     {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    std::vector<std::vector<double>> kernel_x = {
        {-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

    std::vector<std::vector<double>> kernel_y = {
        {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};

    auto vec = image_processing::ImageTo2dVec(image);

    for (auto &vec_el : vec) {
      for (auto &el : vec_el) {
        if (el < 3000) {
          el = 0;
        }
      }
    }

    const auto equalized = image_processing::ImageTo2dVec(
        image_processing::Equalize(image_processing::VecToImage(vec)));

    vec = image_processing::Convolution2D(vec, kernel_x, kernel_y);
    // image_processing::ThresholdOp(vec);

    // image = image_processing::VecToImage(
    //     image_processing::DiffModel(equalized, vec));
    // image = image_processing::VecToImage(vec);
    image = image_processing::VecToImage(equalized);
  } else {
    std::cerr << "Unhandled --type: " << runtime_type << std::endl;
    return 1;
  }

  if (program.is_used("--output")) {
    image.write(program.get<std::string>("--output"));
  } else {
    std::cout << "--output: can be required for some actions" << std::endl;
  }

  return 0;
}
