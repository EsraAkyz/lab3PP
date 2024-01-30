#include "encryption/FES.h"

BitmapImage FES::encrypt(const BitmapImage& original_image, const FES::key_type& encryption_key) {
	const auto height = original_image.get_height();
	const auto width = original_image.get_width();

	auto encrypted_image = BitmapImage{ height, width };

	for (auto y = std::uint32_t(0); y < height; y += 3) {
		for (auto x = std::uint32_t(0); x < width; x += 48) {
			const auto& current_block = get_block(y, x, original_image);
			const auto& previous_block = get_previous_block(y, x, encryption_key, original_image);

			const auto& encrypted_block = encrypt_block(current_block, previous_block);

			set_block(y, x, encrypted_block, encrypted_image);
		}
	}

	return encrypted_image;
}

BitmapImage FES::encrypt_parallel_coarse(const BitmapImage& original_image, const key_type& encryption_key, const int number_of_threads) {
	const auto height = original_image.get_height();
	const auto width = original_image.get_width();

	auto encrypted_image = BitmapImage{ height, width };

#pragma omp parallel for num_threads(number_of_threads)
	for (auto y = std::uint32_t(0); y < height; y += 3) {
		for (auto x = std::uint32_t(0); x < width; x += 48) {
			const auto& current_block = get_block(y, x, original_image);
			const auto& previous_block = get_previous_block(y, x, encryption_key, original_image);

			const auto& encrypted_block = encrypt_block(current_block, previous_block);

			set_block(y, x, encrypted_block, encrypted_image);
		}
	}

	return encrypted_image;
}

BitmapImage FES::encrypt_parallel_fine(const BitmapImage& original_image, const key_type& encryption_key, const int number_of_threads) {
	const auto height = original_image.get_height();
	const auto width = original_image.get_width();

	auto encrypted_image = BitmapImage{ height, width };

	for (auto y = std::uint32_t(0); y < height; y += 3) {
#pragma omp parallel for num_threads(number_of_threads)
		for (auto x = std::uint32_t(0); x < width; x += 48) {
			const auto& current_block = get_block(y, x, original_image);
			const auto& previous_block = get_previous_block(y, x, encryption_key, original_image);

			const auto& encrypted_block = encrypt_block(current_block, previous_block);

			set_block(y, x, encrypted_block, encrypted_image);
		}
	}

	return encrypted_image;
}

BitmapImage FES::encrypt_parallel(const BitmapImage& original_image, const key_type& encryption_key, const int number_of_threads) {
	const auto height = original_image.get_height();
	const auto width = original_image.get_width();

	auto encrypted_image = BitmapImage{ height, width };

#pragma omp parallel for num_threads(number_of_threads) collapse(2)
	for (auto y = std::uint32_t(0); y < height; y += 3) {
		for (auto x = std::uint32_t(0); x < width; x += 48) {
			const auto& current_block = get_block(y, x, original_image);
			const auto& previous_block = get_previous_block(y, x, encryption_key, original_image);

			const auto& encrypted_block = encrypt_block(current_block, previous_block);

			set_block(y, x, encrypted_block, encrypted_image);
		}
	}

	return encrypted_image;
}

BitmapImage FES::decrypt(const BitmapImage& encrypted_image, const key_type& encryption_key) {
	const auto height = encrypted_image.get_height();
	const auto width = encrypted_image.get_width();

	auto decrypted_image = BitmapImage{ height, width };

	for (auto y = std::uint32_t(0); y < height; y += 3) {
		for (auto x = std::uint32_t(0); x < width; x += 48) {
			const auto& current_block = get_block(y, x, encrypted_image);
			const auto& previous_block = get_previous_block(y, x, encryption_key, decrypted_image);

			const auto& decrypted_block = decrypt_block(current_block, previous_block);

			set_block(y, x, decrypted_block, decrypted_image);
		}
	}

	return decrypted_image;
}

FES::block_type FES::encrypt_block(const block_type& current_block, const block_type& previous_block) {
	auto encrypted_block = block_type{};

	for (auto y = std::size_t(0); y < 3; y++) {
		for (auto x = std::size_t(0); x < 3; x++) {
			const auto& current_row = current_block[y][x];
			const auto& previous_row = previous_block[y][x];

			auto working_row = combine_rows(current_row, previous_row);

			for (auto previous_y = std::size_t(0); previous_y < y; previous_y++) {
				for (auto previous_x = std::size_t(0); previous_x < 3; previous_x++) {
					const auto& super_duper_previous_row = encrypted_block[previous_y][previous_x];
					working_row = combine_rows(working_row, super_duper_previous_row);
				}
			}

			for (auto previous_x = std::size_t(0); previous_x < x; previous_x++) {
				const auto& super_duper_previous_row = encrypted_block[y][previous_x];
				working_row = combine_rows(working_row, super_duper_previous_row);
			}

			encrypted_block[y][x] = working_row;
		}
	}

	return encrypted_block;
}

FES::block_type FES::decrypt_block(const block_type& current_block, const block_type& previous_block) {
	auto decrypted_block = block_type{};

	for (auto y = std::size_t(0); y < 3; y++) {
		for (auto x = std::size_t(0); x < 3; x++) {
			const auto& current_row = current_block[y][x];
			const auto& previous_row = previous_block[y][x];

			auto working_row = combine_rows(current_row, previous_row);

			for (auto previous_y = std::size_t(0); previous_y < y; previous_y++) {
				for (auto previous_x = std::size_t(0); previous_x < 3; previous_x++) {
					const auto& super_duper_previous_row = current_block[previous_y][previous_x];
					working_row = combine_rows(working_row, super_duper_previous_row);
				}
			}

			for (auto previous_x = std::size_t(0); previous_x < x; previous_x++) {
				const auto& super_duper_previous_row = current_block[y][previous_x];
				working_row = combine_rows(working_row, super_duper_previous_row);
			}

			decrypted_block[y][x] = working_row;
		}
	}

	return decrypted_block;
}

FES::row_type FES::combine_rows(const row_type& row_1, const row_type& row_2) noexcept {
	auto new_row = row_type{};

	for (auto i = std::size_t(0); i < 16; i++) {
		new_row[i] = row_1[i] ^ row_2[i];
	}

	return new_row;
}

FES::block_type FES::convert_key_to_block(const key_type& key) noexcept {
	auto block = block_type{};

	block[0][0] = convert_key_to_row(key);
	block[0][1] = convert_key_to_row(key);
	block[0][2] = convert_key_to_row(key);

	block[1][0] = convert_key_to_row(key);
	block[1][1] = convert_key_to_row(key);
	block[1][2] = convert_key_to_row(key);

	block[2][0] = convert_key_to_row(key);
	block[2][1] = convert_key_to_row(key);
	block[2][2] = convert_key_to_row(key);

	return block;
}

FES::row_type FES::convert_key_to_row(const key_type& key) noexcept {
	auto row = row_type{};

	for (auto i = std::size_t(0); i < 16; i++) {
		row[i] = { key[3 * i], key[3 * i + 1], key[3 * i + 2] };
	}

	return row;
}

FES::row_type FES::get_row(const std::uint32_t y, const std::uint32_t x, const BitmapImage& image) {
	auto row = row_type{};

	if (y >= image.get_height()) {
		return row;
	}

	for (auto x_in = 0; x + x_in < image.get_width() && x_in < 16; x_in++) {
		row[x_in] = image.get_pixel(y, x + x_in);
	}

	return row;
}

FES::block_type FES::get_block(const std::uint32_t y, const std::uint32_t x, const BitmapImage& image) {
	auto block = block_type{};

	const auto& indices = translate_indices(y, x, image.get_height(), image.get_width());

	block[0][0] = get_row(indices[0].first, indices[0].second, image);
	block[0][1] = get_row(indices[1].first, indices[1].second, image);
	block[0][2] = get_row(indices[2].first, indices[2].second, image);

	block[1][0] = get_row(indices[3].first, indices[3].second, image);
	block[1][1] = get_row(indices[4].first, indices[4].second, image);
	block[1][2] = get_row(indices[5].first, indices[5].second, image);

	block[2][0] = get_row(indices[6].first, indices[6].second, image);
	block[2][1] = get_row(indices[7].first, indices[7].second, image);
	block[2][2] = get_row(indices[8].first, indices[8].second, image);

	return block;
}

FES::block_type FES::get_previous_block(const std::uint32_t y, const std::uint32_t x, const FES::key_type& encryption_key, const BitmapImage& image) {
	if (x == 0) {
		return convert_key_to_block(encryption_key);
	}

	return get_block(y, x - 48, image);
}

void FES::set_row(const std::uint32_t y, const std::uint32_t x, const row_type& row, BitmapImage& image) {
	if (y >= image.get_height()) {
		return;
	}

	for (auto x_in = 0; x + x_in < image.get_width() && x_in < 16; x_in++) {
		image.set_pixel(y, x + x_in, row[x_in]);
	}
}

void FES::set_block(const std::uint32_t y, const std::uint32_t x, const block_type& block, BitmapImage& image) {
	const auto& indices = translate_indices(y, x, image.get_height(), image.get_width());

	set_row(indices[0].first, indices[0].second, block[0][0], image);
	set_row(indices[1].first, indices[1].second, block[0][1], image);
	set_row(indices[2].first, indices[2].second, block[0][2], image);

	set_row(indices[3].first, indices[3].second, block[1][0], image);
	set_row(indices[4].first, indices[4].second, block[1][1], image);
	set_row(indices[5].first, indices[5].second, block[1][2], image);

	set_row(indices[6].first, indices[6].second, block[2][0], image);
	set_row(indices[7].first, indices[7].second, block[2][1], image);
	set_row(indices[8].first, indices[8].second, block[2][2], image);
}
