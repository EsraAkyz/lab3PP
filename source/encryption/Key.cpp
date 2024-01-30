#include "encryption/Key.h"

#include "util/Hash.h"

#include <algorithm>
#include <cstring>
#include <limits>

Key::key_type Key::get_standard_key() noexcept {
	auto key = key_type{};
	for (auto i = key_type::value_type(0); i < key.size(); i++) {
		key[i] = i;
	}

	return key;
}

Key::key_type Key::produce_new_key(const key_type& old_key) noexcept {
	auto new_key = key_type{};

	for (auto i = std::size_t(0); i < 6; i++) {
		auto val = std::uint64_t{ 0 };
		std::memcpy(&val, old_key.data() + i * 8, 8);

		auto other_val = Hash::hash(val);
		std::memcpy(new_key.data() + i * 8, &other_val, 8);
	}

	return new_key;
}

std::uint64_t Key::hash(const key_type& key) noexcept {
	auto hash_value = std::uint64_t{ 0 };

	for (auto i = std::size_t(0); i < 6; i++) {
		auto val = std::uint64_t{ 0 };
		std::memcpy(&val, key.data() + i * 8, 8);

		auto other_val = Hash::hash(val);
		hash_value = Hash::combine_hashes(hash_value, other_val);
	}

	return hash_value;
}

std::uint64_t Key::get_smallest_hash(const std::span<const key_type> values) noexcept {
	auto smallest_hash = std::numeric_limits<std::uint64_t>::max();

	for (const auto& key : values) {
		const auto hash_value = hash(key);
		smallest_hash = std::min(smallest_hash, hash_value);
	}

	return smallest_hash;
}

std::uint64_t Key::get_smallest_hash_parallel(std::span<const key_type> values, int number_threads) noexcept {
	auto smallest_hash = std::numeric_limits<std::uint64_t>::max();

#pragma omp parallel for reduction(min:smallest_hash) num_threads(number_threads)
	for (auto i = 0; i < values.size(); i++) {
		const auto& key = values[i];
		const auto hash_value = hash(key);
		smallest_hash = std::min(smallest_hash, hash_value);
	}

	return smallest_hash;
}

Key::key_type Key::find_key(std::span<const key_type> values, std::uint64_t hash_value) noexcept {
	for (const auto& value : values) {
		if (hash(value) == hash_value) {
			return value;
		}
	}

	return {};
}

Key::key_type Key::find_key_parallel(std::span<const key_type> values, std::uint64_t hash_value, int number_threads) noexcept {
	auto found_key = key_type{};

#pragma omp parallel for num_threads(number_threads)
	for (auto i = 0; i < values.size(); i++) {
		const auto& key = values[i];
		const auto calculated_hash = hash(key);

		if (hash_value == calculated_hash) {
#pragma omp critical
			found_key = key;
		}
	}

	return found_key;
}
