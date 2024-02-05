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
	using bit = std::array<int, 64>;

	// task 1 b)
	[[nodiscard]] std::uint64_t encode(EncryptionScheme encryptionScheme);

	// task 1 c)
	[[nodiscard]] EncryptionScheme decode(std::uint64_t i);

	// task 1 d)
	[[nodiscard]] BitmapImage perform_scheme(BitmapImage bmi, Key::key_type keytype, EncryptionScheme e_scheme);

	// task 3 e)
	/** retrieve_scheme function **/
	EncryptionScheme retrieve_scheme(std::uint64_t c){
    		EncryptionScheme result;

   		 // Iterate over all possible scheme
    		// 1ULL is the integer literal 1 (ULL stands for "Unsigned Long Long"), represented as an unsigned 64-bit integer
    		// (1ULL << 20) equals 2^20
    		for(auto i = std::uint64_t(0); i < (1ULL << 20); i++){
        		std::uint64_t encoded_scheme = encode(i);

		        // Check if the hash matches encoded scheme
        		if(c == encoded_scheme){
            			result = decode(i);

            			// Get standard key for first encryption
            			Key::key_type encryption_key = Key::get_standard_key();
 	           		BitmapImage = image;

       	   			int step = 0;
      		      		// First 10 steps
        			while (step < 10) {
                			image = perform_scheme(result, encryption_key, encoded_scheme);
                			step++;
            			}

		        	// Next 6 steps
            			while (step < 16) {
					encode(decoded_scheme[step]);
					step++;
				}

		        	break;
        		}
    		}
   		return result;
	}

};
