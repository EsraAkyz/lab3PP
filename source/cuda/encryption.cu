#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// task 3 a)
__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {

	constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };          // Threadblöcke nicht beachtet
	constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
	constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

	for (auto i = 0; i < length; i++) {
		const auto val_1 = (values[i] >> 14) + val_a;
		const auto val_2 = (values[i] << 54) ^ val_b;
		const auto val_3 = (val_1 + val_2) << 4;
		const auto val_4 = (values[i] % val_c) * 137;
		
		hashes[i] = val_3 ^ val_4;
	}
	
}
// task 3 b)
__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){

	constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };          // Threadblöcke nicht beachtet
	constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
	constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

	for (auto i = 0; i < length; i++) {
		const auto val_1 = (values[i] >> 14) + val_a;
		const auto val_2 = (values[i] << 54) ^ val_b;
		const auto val_3 = (val_1 + val_2) << 4;
		const auto val_4 = (values[i] % val_c) * 137;

		hashes[i] = val_3 ^ val_4;
	}
}

// task 3 c)
__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr) {
	int index = 0;
	for (auto i = 0; i < length; i++) {
		if (searched_hash == hashes[i]) {
			indices[index] = i;                                                // Threadblöcke nicht beachtet
			index++; 
		}
	}
}

// task 3 d)
__global__ void hash_schemes(std::uint64_t* const hashes, const unsigned int length) {
	for (auto i = 0; i < length; i++) {
		hashes[i]= 
	}

}