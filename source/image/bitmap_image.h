#pragma once

#include "image/pixel.h"

#include <cstdint>
#include <vector>

class BitmapImage {
public:
	using BitmapPixel = Pixel<std::uint8_t>;
	using index_type = std::uint32_t;

	BitmapImage(const std::uint32_t image_height, const std::uint32_t image_width);

	void set_pixel(const std::uint32_t y_position, const std::uint32_t x_position, const BitmapPixel pixel);

	[[nodiscard]] BitmapPixel get_pixel(const std::uint32_t y_position, const std::uint32_t x_position) const;

	[[nodiscard]] std::uint32_t get_height() const noexcept;

	[[nodiscard]] std::uint32_t get_width() const noexcept;

	[[nodiscard]] BitmapImage transpose() const;

	[[nodiscard]] BitmapImage get_grayscale() const;

	void mystery();

	[[nodiscard]] BitmapImage transpose_parallel(int number_of_threads = 1) const;

	[[nodiscard]] BitmapImage get_grayscale_parallel(int number_of_threads = 1) const;

	void mystery_parallel(int number_of_threads = 1);

	[[nodiscard]] const BitmapPixel* get_data() const noexcept {
		return pixels.data();
	}

	[[nodiscard]] BitmapPixel* get_data() noexcept {
		return pixels.data();
	}

private:
	std::uint32_t height{};
	std::uint32_t width{};

	std::vector<BitmapPixel> pixels{};
};
