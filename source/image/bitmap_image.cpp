#include "image/bitmap_image.h"

#include <exception>

BitmapImage::BitmapImage(const std::uint32_t image_height, const std::uint32_t image_width)
	: height{ image_height }, width{ image_width } {
	if (image_height > 1'024'000) {
		throw std::exception{ };
	}

	if (image_height == 0) {
		throw std::exception{ };
	}

	if (image_width > 1'024'000) {
		throw std::exception{ };
	}

	if (image_width == 0) {
		throw std::exception{  };
	}

	pixels.resize(image_height * image_width, BitmapPixel{ 0, 0, 0 });
}

void BitmapImage::set_pixel(const std::uint32_t y_position, const std::uint32_t x_position, const BitmapPixel pixel) {
	if (y_position >= height) {
		throw std::exception{  };
	}

	if (x_position >= width) {
		throw std::exception{ };
	}

	pixels[y_position * width + x_position] = pixel;
}

BitmapImage::BitmapPixel BitmapImage::get_pixel(const std::uint32_t y_position, const std::uint32_t x_position) const {
	if (y_position >= height) {
		throw std::exception{ };
	}

	if (x_position >= width) {
		throw std::exception{ };
	}

	return pixels[y_position * width + x_position];
}

std::uint32_t BitmapImage::get_height() const noexcept {
	return height;
}

std::uint32_t BitmapImage::get_width() const noexcept {
	return width;
}

BitmapImage BitmapImage::transpose() const {
	auto transposed_image = BitmapImage(width, height);

	for (auto y = std::uint32_t(0); y < height; y++) {
		for (auto x = std::uint32_t(0); x < width; x++) {
			transposed_image.set_pixel(x, y, get_pixel(y, x));
		}
	}

	return transposed_image;
}

BitmapImage BitmapImage::get_grayscale() const {
	auto gray_image = BitmapImage(height, width);

	for (auto y = std::uint32_t(0); y < height; y++) {
		for (auto x = std::uint32_t(0); x < width; x++) {
			const auto pixel = get_pixel(y, x);
			const auto r = pixel.get_red_channel();
			const auto g = pixel.get_green_channel();
			const auto b = pixel.get_blue_channel();

			const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
			const auto gray_converted = static_cast<std::uint8_t>(gray);

			const auto gray_pixel = BitmapPixel{ gray_converted , gray_converted,  gray_converted };

			gray_image.set_pixel(y, x, gray_pixel);
		}
	}

	return gray_image;
}

void BitmapImage::mystery() {
	const auto construct_pixel = [](const std::uint8_t red, const std::uint8_t green, const std::uint8_t blue) -> BitmapPixel {
		switch (BitmapImage::BitmapPixel::channel_order) {
		case ChannelOrder::BGR:
			return { blue, green, red };
		case ChannelOrder::BRG:
			return { blue, red, green };
		case ChannelOrder::GBR:
			return { green, blue, red };
		case ChannelOrder::GRB:
			return { green, red, blue };
		case ChannelOrder::RBG:
			return { red, blue, green };
		case ChannelOrder::RGB:
			return { red, green, blue };
		}

		throw std::exception{};
		};

	for (auto y = std::uint32_t(0); y < height; y++) {
		for (auto x = std::uint32_t(0); x < width; x++) {
			const auto smallest_x = x;
			const auto largest_x = (x + 2 < width) ? x + 2 : width;

			const auto smallest_y = y;
			const auto largest_y = (y + 2 < height) ? y + 2 : height;

			auto red_acc = std::uint16_t{ 0 };
			auto green_acc = std::uint16_t{ 0 };
			auto blue_acc = std::uint16_t{ 0 };

			for (auto y_it = smallest_y; y_it < largest_y; y_it++) {
				for (auto x_it = smallest_x; x_it < largest_x; x_it++) {
					const auto pixel = get_pixel(y_it, x_it);
					const auto r = pixel.get_red_channel();
					const auto g = pixel.get_green_channel();
					const auto b = pixel.get_blue_channel();

					red_acc += r;
					green_acc += g;
					blue_acc += b;
				}
			}

			const auto red_cut = static_cast<std::uint8_t>(red_acc);
			const auto green_cut = static_cast<std::uint8_t>(green_acc);
			const auto blue_cut = static_cast<std::uint8_t>(blue_acc);

			const auto new_pixel = construct_pixel(red_cut, green_cut, blue_cut);

			set_pixel(y, x, new_pixel);
		}
	}
}

BitmapImage BitmapImage::transpose_parallel(int number_of_threads) const {
	auto transposed_image = BitmapImage(width, height);

	const auto _height = height;
	const auto _width = width;

#pragma omp parallel for num_threads(number_of_threads) collapse(2)
	for (auto y = std::uint32_t(0); y < _height; y++) {
		for (auto x = std::uint32_t(0); x < _width; x++) {
			transposed_image.set_pixel(x, y, get_pixel(y, x));
		}
	}

	return transposed_image;
}

BitmapImage BitmapImage::get_grayscale_parallel(int number_of_threads) const {
	auto gray_image = BitmapImage(height, width);

	const auto _height = height;
	const auto _width = width;

#pragma omp parallel for num_threads(number_of_threads) collapse(2)
	for (auto y = std::uint32_t(0); y < _height; y++) {
		for (auto x = std::uint32_t(0); x < _width; x++) {
			const auto pixel = get_pixel(y, x);
			const auto r = pixel.get_red_channel();
			const auto g = pixel.get_green_channel();
			const auto b = pixel.get_blue_channel();

			const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
			const auto gray_converted = static_cast<std::uint8_t>(gray);

			const auto gray_pixel = BitmapPixel{ gray_converted , gray_converted,  gray_converted };

			gray_image.set_pixel(y, x, gray_pixel);
		}
	}

	return gray_image;
}

void BitmapImage::mystery_parallel(int number_of_threads) {
	const auto construct_pixel = [](const std::uint8_t red, const std::uint8_t green, const std::uint8_t blue) -> BitmapPixel {
		switch (BitmapImage::BitmapPixel::channel_order) {
		case ChannelOrder::BGR:
			return { blue, green, red };
		case ChannelOrder::BRG:
			return { blue, red, green };
		case ChannelOrder::GBR:
			return { green, blue, red };
		case ChannelOrder::GRB:
			return { green, red, blue };
		case ChannelOrder::RBG:
			return { red, blue, green };
		case ChannelOrder::RGB:
			return { red, green, blue };
		}

		throw std::exception{};
		};

	auto copy = *this;

	const auto _height = height;
	const auto _width = width;

#pragma omp parallel for num_threads(number_of_threads) collapse(2)
	for (auto y = std::uint32_t(0); y < _height; y++) {
		for (auto x = std::uint32_t(0); x < _width; x++) {
			const auto smallest_x = x;
			const auto largest_x = (x + 2 < _width) ? x + 2 : _width;

			const auto smallest_y = y;
			const auto largest_y = (y + 2 < _height) ? y + 2 : _height;

			auto red_acc = std::uint16_t{ 0 };
			auto green_acc = std::uint16_t{ 0 };
			auto blue_acc = std::uint16_t{ 0 };

			for (auto y_it = smallest_y; y_it < largest_y; y_it++) {
				for (auto x_it = smallest_x; x_it < largest_x; x_it++) {
					const auto pixel = copy.get_pixel(y_it, x_it);
					const auto r = pixel.get_red_channel();
					const auto g = pixel.get_green_channel();
					const auto b = pixel.get_blue_channel();

					red_acc += r;
					green_acc += g;
					blue_acc += b;
				}
			}

			const auto red_cut = static_cast<std::uint8_t>(red_acc);
			const auto green_cut = static_cast<std::uint8_t>(green_acc);
			const auto blue_cut = static_cast<std::uint8_t>(blue_acc);

			const auto new_pixel = construct_pixel(red_cut, green_cut, blue_cut);

			set_pixel(y, x, new_pixel);
		}
	}
}
