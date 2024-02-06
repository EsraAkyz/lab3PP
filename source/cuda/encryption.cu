#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "util/Hash.h"
#include <encryption/Algorithm.h>
#include <device_functions.h>
#include "cuda/common.cuh"

// task 3 a)
__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {

	/** Kernel of hash function **/

		// Calculate the global index of the current thread
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if thread is within length
	if (index < length) {
		constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
		constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
		constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

		const auto val_1 = (values[index] >> 14) + val_a;
		const auto val_2 = (values[index] << 54) ^ val_b;
		const auto val_3 = (val_1 + val_2) << 4;
		const auto val_4 = (values[index] % val_c) * 137;

		const auto final_hash = val_3 ^ val_4;
		hashes[index] = final_hash;
	}
}

// task 3 b)
/** Kernel of flat_hash **/
__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {
	// Allocate shared memory
	__shared__ std::uint64_t shared_values[FLAT_HASH_SHARED_MEM];
	__shared__ std::uint64_t shared_hashes[FLAT_HASH_SHARED_MEM];

	// Calculate the global index of current thread
	 // Cause the kernel is invoked with (1, 1, 1) thread blocks of size (tx, 1, 1)

	// Check if thread within length
	
	//hashes[i] = final_hash

	if (threadIdx.x < length) {
		unsigned int tx = threadIdx.x;
		// Load the value into shared memory
		shared_values[tx] = values[tx];

		// Calculate the hash value for the corresponding value
		constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
		constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
		constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

		const auto val_1 = (shared_values[tx] >> 14) + val_a;
		const auto val_2 = (shared_values[tx] << 54) ^ val_b;
		const auto val_3 = (val_1 + val_2) << 4;
		const auto val_4 = (shared_values[tx] % val_c) * 137;

		const auto final_hash = val_3 ^ val_4;

		// Write the hash value to global memory
		hashes[tx] = final_hash;
	}

}


// 3 c)

/** Kernel for find_hash **/
__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr) {
	// Calculate the global index of the current thread
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// Shared memory declaration
	__shared__ unsigned int shared_indices[FIND_HASH_SHARED_MEM];

	// Initialize shared memory
	for (auto i = threadIdx.x; i < FIND_HASH_SHARED_MEM; i += blockDim.x) {
		shared_indices[i] = 0;
	}

	// Search for the hash in the given range
	for (auto i = index; i < blockIdx.x * blockIdx.y;i++) {
		if (hashes[i] == searched_hash) {
			// Atomically update the shared index array
			auto position = atomicAdd(ptr, 1);
			shared_indices[position % FIND_HASH_SHARED_MEM] = i;
		}
	}


	// Copy shared indices to global memory
	for (unsigned int i = threadIdx.x; i < FIND_HASH_SHARED_MEM; i += blockDim.x) {
		if (shared_indices[i] > 0) {
			auto position = atomicAdd(ptr, 1);
			indices[position] = shared_indices[i];
		}
	}
}

// 3 d)

/** Kernel for hash_scheme **/
__global__ void hash_schemes(std::uint64_t* const hashes, const unsigned int length) {
	// Declare shared memory
	__shared__ std::uint64_t shared_hashes[HASH_SCHEMES_SHARED_MEM];

	// Calculate the global index of the current thread
	auto index = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if thread within length
	if (index < length) {
		// Convert index to an EncryptionScheme
		EncryptionScheme scheme =  decode(index);

		// Convert EncryptionScheme to a std::uint64_t value for hashing
		std::uint64_t combined_scheme = encode(scheme);

		unsigned int tx = threadIdx.x;

		// Hash value for combined scheme
		constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
		constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
		constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };

		const auto val_1 = (combined_scheme >> 14) + val_a;
		const auto val_2 = (combined_scheme << 54) ^ val_b;
		const auto val_3 = (val_1 + val_2) << 4;
		const auto val_4 = (combined_scheme % val_c) * 137;

		const auto final_hash = val_3 ^ val_4;


		// Copy shared hashes to global memory
		hashes[index] = final_hash; 
	}
} 
