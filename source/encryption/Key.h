#pragma once

#include <array>
#include <cstdint>
#include <span>

class Key {
public:
	using key_type = std::array<std::uint8_t, 48>;

	[[nodiscard]] static key_type get_standard_key() noexcept;

	[[nodiscard]] static key_type produce_new_key(const key_type& old_key) noexcept;

	[[nodiscard]] static std::uint64_t hash(const key_type& key) noexcept;

	[[nodiscard]] static std::uint64_t get_smallest_hash(std::span<const key_type> values) noexcept;

	[[nodiscard]] static std::uint64_t get_smallest_hash_parallel(std::span<const key_type> values, int number_threads = 1) noexcept;

	[[nodiscard]] static key_type find_key(std::span<const key_type> values, std::uint64_t hash_value) noexcept;

	[[nodiscard]] static key_type find_key_parallel(std::span<const key_type> values, std::uint64_t hash_value, int number_threads = 1) noexcept;

private:

};
