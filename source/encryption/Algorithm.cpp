#include "Algorithm.h"
#include "FES.h"

// task 1 b)
/** encode function **/
std::uint64_t encode(const Algorithm::EncryptionScheme& scheme) {
	std::uint64_t result = 0;
	std::uint64_t bits;
	// Loop through each step in the scheme
	for (auto i = 0; i < scheme.size(); i++) {
		// Translate EncryptionStep into bits
		switch (scheme[i]) {
			case EncryptionStep::E: bits = 0b00; break;
			case EncryptionStep::D: bits = 0b01; break;
			case EncryptionStep::K: bits = 0b10; break;
			case EncryptionStep::T: bits = 0b11; break;
			default: bits = 0; // Handle potential errors
		}
		// Arrange bits in std::uint64_t
		result |= (bits << (i * 2));
	}

	// Replicate the 32 lower bits into the 32 upper bits
	result |= (result << 32);

	return result;
}

// task 1 c)
[[nodiscard]] Algorithm::EncryptionScheme decode(std::uint64_t number){
	/* Algorithm::EncryptionScheme enScheme = Algorithm::EncryptionScheme{};
	auto copyArray = Algorithm::bit{};
	auto firstBit = 0;
	auto secondBit = 0;
	int indexScheme = 0;

	// copy i to copyArray
	for (auto n = 63; n > 0; i--) {
		copyArray[n] = (i >> i) & 1; 
	}

	// compare index 0-31 and 32-63
	for(auto j=0; j<32 ; j++){
		if (copyArray[i] != copyArray[i + 32])
			throw std::exception{};
	}

	// fill EncryptionSCheme with associated values
	for (auto k = 0; k < 64; k=k+2) {
		firstBit = copyArray[k];
		secondBit = copyArray[k + 1];

		if (firstBit == 0 && secondBit == 0 ) {
			enScheme[indexScheme] = EncryptionStep::E;
		}
		if (firstBit == 0 && secondBit == 1 ) {
			enScheme[indexScheme] = EncryptionStep::D;
		}
		if (firstBit == 1 && secondBit == 0 ) {
			enScheme[indexScheme] = EncryptionStep::K;
		}
		else {
			enScheme[indexScheme] = EncryptionStep::T;
		}
		indexScheme++; 
	}
	return enScheme;*/
	//task 1c          
	
		if ((number& 0xFFFFFFFF) != ((number >> 32) & 0xFFFFFFFF)) {
			throw std::exception();
		}

		Algorithm::EncryptionScheme scheme{};
		for (int i = 0; i < scheme.size(); i++) {
			std::uint64_t bitPair = (number >> (i * 2)) & 0b11;
			switch (bitPair)
			{
			case 0b00:
				scheme[i] = EncryptionStep::E;
				break;
			case 0b01:
				scheme[i] = EncryptionStep::D;
				break;
			case 0b10:
				scheme[i] = EncryptionStep::K;
				break;
			case 0b11:
				scheme[i] = EncryptionStep::T;
				break;
			}
		}
		return scheme;
	
}

// task 1 d)
[[nodiscard]] BitmapImage perform_scheme( BitmapImage bitmap, Key::key_type keytype, Algorithm::EncryptionScheme e_scheme){ 
	
	for (auto i = 0; i < 16; i++) {
		if (e_scheme[i] == EncryptionStep::E)
			bitmap = FES::encrypt(bitmap, keytype);
		if (e_scheme[i] == EncryptionStep::D)
			bitmap= FES::decrypt(bitmap, keytype);
		if (e_scheme[i] == EncryptionStep::T)
			bitmap =bitmap.transpose();
		if (e_scheme[i] == EncryptionStep::K)
			keytype = Key::produce_new_key(keytype);
	}
	return bitmap;
}

// 3 e)
[[nodiscard]] Algorithm::EncryptionScheme retrieve_scheme(std::uint64_t c) {
	Algorithm::EncryptionScheme result;
	/*
	// Iterate over all possible scheme
	   // 1ULL is the integer literal 1 (ULL stands for "Unsigned Long Long"), represented as an unsigned 64-bit integer
	   // (1ULL << 20) equals 2^20
	for (auto i = std::uint64_t(0); i < (1ULL << 20); i++) {
		Algorithm::EncryptionScheme es = Algorithm::decode(c);
		std::uint64_t encoded_scheme = encode();

		// Check if the hash matches encoded scheme
		if (c == encoded_scheme) {
			result = decode(i);

			// Get standard key for first encryption
			Key::key_type encryption_key = Key::get_standard_key();
			BitmapImage image();
			BitmapImage* p = nullptr;

			int step = 0;
			// First 10 steps
			while (step < 10) {
				image = perform_scheme(image, encryption_key, encoded_scheme);
				step++;
			}

			// Next 6 steps
			while (step < 16) {
				encode(decoded_scheme[step]);
				step++;
			}

			break;
		}
	} */
	return result;
}
