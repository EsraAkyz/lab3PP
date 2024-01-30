#pragma once

#include <cstdint>

class Hash {
public:
	using hash_type = std::uint64_t;

	[[nodiscard]] static hash_type hash(std::uint64_t value) noexcept;

	[[nodiscard]] static hash_type hash(std::uint32_t value) noexcept;

	[[nodiscard]] static hash_type hash(std::uint16_t value) noexcept;

	[[nodiscard]] static hash_type hash(std::uint8_t value) noexcept;

	[[nodiscard]] static hash_type combine_hashes(hash_type first_hash, hash_type second_hash) noexcept;

private:

};
