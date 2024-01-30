#include "util/Hash.h"

Hash::hash_type Hash::hash(const std::uint64_t value) noexcept {
	constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
	constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
	constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

	const auto val_1 = (value >> 14) + val_a;
	const auto val_2 = (value << 54) ^ val_b;
	const auto val_3 = (val_1 + val_2) << 4;
	const auto val_4 = (value % val_c) * 137;

	const auto final_hash = val_3 ^ val_4;
	return final_hash;
}

Hash::hash_type Hash::hash(const std::uint32_t value) noexcept {
	const auto promoted = static_cast<std::uint64_t>(value);
	return hash(promoted);
}

Hash::hash_type Hash::hash(const std::uint16_t value) noexcept {
	const auto promoted = static_cast<std::uint64_t>(value);
	return hash(promoted);
}

Hash::hash_type Hash::hash(const std::uint8_t value) noexcept {
	const auto promoted = static_cast<std::uint64_t>(value);
	return hash(promoted);
}

Hash::hash_type Hash::combine_hashes(const hash_type first_hash, const hash_type second_hash) noexcept {
	return first_hash ^ second_hash;
}
