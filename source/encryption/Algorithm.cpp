#include "Algorithm.h"
#include "FES.h"

// task 1 b)
[[nodiscard]] std::uint64_t encode(Algorithm::EncryptionScheme encryptionScheme) {

	auto first = 0;
	auto second = 0; 
	std::uint64_t result; 
	std::vector<int> values;
	auto newArray = Algorithm::bit{};

	for (auto i = 0; i < 31; i=i+2) {
		if (encryptionScheme[i] == EncryptionStep::E) {
			first = 0;
			second = 0;
		}
		if (encryptionScheme[i] == EncryptionStep::D) {
			first = 0;
			second = 1;
		}
		if (encryptionScheme[i] == EncryptionStep::K) {
			first = 1;
			second = 0;
		}
		else {
			first = 1;
			second = 1;
		}

		newArray[31 - i] = second;
		newArray[30 - i] = first;
		
	}
	for (auto j = 0; j < 32; j++) {
		newArray[32 + j] = newArray[j];
	}

	std::memcpy(&result, &newArray, 64);

	return result;
}

// task 1 c)
[[nodiscard]] Algorithm::EncryptionScheme decode(std::uint64_t i){
	Algorithm::EncryptionScheme enScheme = Algorithm::EncryptionScheme{};
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
	return enScheme;

}

// task 1 d)
[[nodiscard]] BitmapImage perform_scheme( const BitmapImage& bitmap, const Key::key_type& keytype, Algorithm::EncryptionScheme e_scheme){  // const ?
	
	for (auto i = 0; i < 16; i++) {
		if (e_scheme[i] == EncryptionStep::E)
			FES::encrypt(bitmap, keytype);
		if (e_scheme[i] == EncryptionStep::D)
			FES::decrypt(bitmap, keytype);
		if (e_scheme[i] == EncryptionStep::T)
			bitmap.transpose();
		else
			Key::produce_new_key(keytype);

	}
}