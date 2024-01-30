#include "Algorithm.h"

// task 1 b)
[[nodiscard]] std::uint64_t encode(Algorithm::EncryptionScheme encryptionScheme) {
	auto bit = 0;
	std::uint64_t result; 

	for (auto i = 0; i < 16; i++) {
		if (encryptionScheme[i] == EncryptionStep::E) {
			bit = 00;
		}
		if (encryptionScheme[i] == EncryptionStep::D) {
			bit = 01;
		}
		if (encryptionScheme[i] == EncryptionStep::K) {
			bit = 10;
		}
		else {
			bit = 11;
		}
	}
}

// task 1 c)
[[nodiscard]] Algorithm::EncryptionScheme decode(std::uint64_t i){}

// task 1 d)
[[nodiscard]] BitmapImage perform_scheme(BitmapImage bmi, Key::key_type keytype, Algorithm::EncryptionScheme e_scheme){}