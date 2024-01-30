#pragma once

#include "encryption/Key.h"
#include "image/bitmap_image.h"

#include <array>
#include <cstdint>
#include <utility>

class FES {
public:
	using key_type = Key::key_type;
	using row_type = std::array<BitmapImage::BitmapPixel, 16>;
	using block_type = std::array<std::array<row_type, 3>, 3>;

	[[nodiscard]] static BitmapImage encrypt(const BitmapImage& original_image, const key_type& encryption_key);

	[[nodiscard]] static BitmapImage encrypt_parallel_coarse(const BitmapImage& original_image, const key_type& encryption_key, int number_of_threads = 1);

	[[nodiscard]] static BitmapImage encrypt_parallel_fine(const BitmapImage& original_image, const key_type& encryption_key, int number_of_threads = 1);

	[[nodiscard]] static BitmapImage encrypt_parallel(const BitmapImage& original_image, const key_type& encryption_key, int number_of_threads = 1);

	[[nodiscard]] static BitmapImage decrypt(const BitmapImage& encrypted_image, const key_type& encryption_key);

	[[nodiscard]] static block_type encrypt_block(const block_type& current_block, const block_type& previous_block);

	[[nodiscard]] static block_type decrypt_block(const block_type& current_block, const block_type& previous_block);

	[[nodiscard]] static row_type combine_rows(const row_type& row_1, const row_type& row_2) noexcept;

	[[nodiscard]] static block_type convert_key_to_block(const key_type& key) noexcept;

	[[nodiscard]] static row_type convert_key_to_row(const key_type& key) noexcept;

private:
	using index_type = std::pair<std::size_t, std::size_t>;
	using indices_type = std::array<index_type, 9>;

	[[nodiscard]] static row_type get_row(const std::uint32_t y, const std::uint32_t x, const BitmapImage& image);

	[[nodiscard]] static block_type get_block(const std::uint32_t y, const std::uint32_t x, const BitmapImage& image);

	[[nodiscard]] static block_type get_previous_block(const std::uint32_t y, const std::uint32_t x,
		const FES::key_type& encryption_key, const BitmapImage& image);

	static void set_row(const std::uint32_t y, const std::uint32_t x, const row_type& row, BitmapImage& image);

	static void set_block(const std::uint32_t y, const std::uint32_t x, const block_type& block, BitmapImage& image);

	static indices_type translate_indices(const std::size_t y, const std::size_t x, const std::uint32_t height, const std::uint32_t width) noexcept {
		auto indices = indices_type{};

		if (x + 48 <= width) {
			// Each row can be fully used. Later rows are irrelevant
			indices[0] = { index_type::first_type(y),  index_type::second_type(x) };
			indices[1] = { index_type::first_type(y),  index_type::second_type(x + 16) };
			indices[2] = { index_type::first_type(y),  index_type::second_type(x + 32) };

			indices[3] = { index_type::first_type(y + 1),  index_type::second_type(x) };
			indices[4] = { index_type::first_type(y + 1),  index_type::second_type(x + 16) };
			indices[5] = { index_type::first_type(y + 1),  index_type::second_type(x + 32) };

			indices[6] = { index_type::first_type(y + 2),  index_type::second_type(x) };
			indices[7] = { index_type::first_type(y + 2),  index_type::second_type(x + 16) };
			indices[8] = { index_type::first_type(y + 2),  index_type::second_type(x + 32) };
		}
		else if (y + 3 <= height) {
			// Transpose block
			indices[0] = { index_type::first_type(y),  index_type::second_type(x) };
			indices[1] = { index_type::first_type(y + 1),  index_type::second_type(x) };
			indices[2] = { index_type::first_type(y + 2),  index_type::second_type(x) };

			indices[3] = { index_type::first_type(y),  index_type::second_type(x + 16) };
			indices[4] = { index_type::first_type(y + 1),  index_type::second_type(x + 16) };
			indices[5] = { index_type::first_type(y + 2),  index_type::second_type(x + 16) };

			indices[6] = { index_type::first_type(y),  index_type::second_type(x + 32) };
			indices[7] = { index_type::first_type(y + 1),  index_type::second_type(x + 32) };
			indices[8] = { index_type::first_type(y + 2),  index_type::second_type(x + 32) };
		}
		else if (x + 32 <= width) {
			// Only two row blocks are within the picture
			indices[0] = { index_type::first_type(y),  index_type::second_type(x) };
			indices[1] = { index_type::first_type(y),  index_type::second_type(x + 16) };

			indices[2] = { index_type::first_type(y + 1),  index_type::second_type(x) };
			indices[3] = { index_type::first_type(y + 1),  index_type::second_type(x + 16) };

			indices[4] = { index_type::first_type(y + 2),  index_type::second_type(x) };
			indices[5] = { index_type::first_type(y + 2),  index_type::second_type(x + 16) };

			indices[6] = { index_type::first_type(y),  index_type::second_type(x + 32) };
			indices[7] = { index_type::first_type(y + 1),  index_type::second_type(x + 32) };
			indices[8] = { index_type::first_type(y + 2),  index_type::second_type(x + 32) };
		}
		else {
			indices[0] = { index_type::first_type(y),  index_type::second_type(x) };
			indices[1] = { index_type::first_type(y + 1),  index_type::second_type(x) };

			indices[2] = { index_type::first_type(y),  index_type::second_type(x + 16) };
			indices[3] = { index_type::first_type(y + 1),  index_type::second_type(x + 16) };

			indices[4] = { index_type::first_type(y),  index_type::second_type(x + 32) };
			indices[5] = { index_type::first_type(y + 1),  index_type::second_type(x + 32) };

			indices[6] = { index_type::first_type(y + 2),  index_type::second_type(x) };
			indices[7] = { index_type::first_type(y + 2),  index_type::second_type(x + 16) };
			indices[8] = { index_type::first_type(y + 2),  index_type::second_type(x + 32) };
		}

		return indices;
	}
};
