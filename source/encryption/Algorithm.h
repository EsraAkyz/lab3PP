#pragma once
#include <array>
#include <image/bitmap_image.h>
#include "Key.h"

//task 1 a)
enum class EncryptionStep {
	E,
	D,
	K,
	T
};

class Algorithm {
public: 
	using EncryptionScheme = std::array<EncryptionStep, 16>;

	// task 1 b)
	[[nodiscard]] std::uint64_t encode(EncryptionScheme encryptionScheme);

	// task 1 c)
	[[nodiscard]] EncryptionScheme decode(std::uint64_t i);

	// task 1 d)
	[[nodiscard]] BitmapImage perform_scheme(BitmapImage bmi, Key::key_type keytype, EncryptionScheme e_scheme);

};