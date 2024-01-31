#include "Algorithm.h"
#include "FES.h"

// task 1 b)
/** encode function **/
std::uint64_t encode (const EncryptionScheme& scheme) {
    std::uint64_t result = 0;

    // Loop through each step in the scheme
    for (auto i=0; i<scheme.size(); i++){
        // Translate EncryptionStep into bits
        std::uint64_t bits;
        switch (scheme[i]) {
            case E: bits = 0b00; break;
            case D: bits = 0b01; break;
            case K: bits = 0b10; break;
            case T: bits = 0b11; break;
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
/** decode function **/
EncryptionScheme decode (std::uint64_t decodeValue) {
    // Check the form
    if ((decodeValue & 0xFFFFFFFF) != ((decodeValue >> 32) & 0xFFFFFFFF)) {
        throw std::exception();
    }

    EncryptionScheme result;

    // Loop through each steps in scheme
    for (auto i=0; i<result.size(); i++){
        // Extract bits
        std::uint64_t bits = (decodeValue >> (i*2)) & 0b11;

        // Translate bits into EncryptionSteps
        switch (bits) {
            case 0b00: result[i] = E; break;
            case 0b01: result[i] = D; break;
            case 0b10: result[i] = K; break;
            case 0b11: result[i] = T; break;
            default: throw std::exception();
        }
    }

    return result;
}

// task 1 d)
/** perform_scheme function **/
BitmapImage perform_scheme (BitmapImage image, Key::key_type& encryption_key, const EncryptionScheme scheme){
    // Iteration over scheme
    for (auto i=0; i<scheme.size(); i++){
        switch (scheme[i]) {
            case E:
                image = FES::encrypt(image, encryption_key);
                break;
            case D:
                image = FES::descrypt(image, encryption_key);
                break;
            case T:
                image.transpose();
                break;
            case K:
                encryption_key = Key::produce_new_key(encryption_key);
                break;
            default: std::exception();
        }
    }

    return image;
}
