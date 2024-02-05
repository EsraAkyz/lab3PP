#include "encryption.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "util/Hash.h"
#include <encryption/Algorithm.h>
#include <device_functions.h>

// task 3 a)
__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length) {

	/** Kernel of hash function **/
	
		// Calculate the global index of the current thread
		unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

		// Check if thread is within length
		if(index < length) {
			std::uint64_t hash_value = Hash::hash(values[index]);
			hashes[index] = hash_value;
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
	if(threadIdx.x < length) {
		// Load the value into shared memory
		shared_values[threadIdx.x] = values[threadIdx.x];
		//__syncthreads();  // Ensure all threads have loaded their values

		// Calculate the hash value for the corresponding value
		shared_hashes[threadIdx.x] = Hash::hash(shared_values[threadIdx.x]);
		//__syncthreads();  // Ensure all threads have calculated their hashes

		// Write the hash value to global memory
		hashes[threadIdx.x] = shared_hashes[threadIdx.x];
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
	for(auto i = threadIdx.x; i < FIND_HASH_SHARED_MEM; i += blockDim.x) {
		shared_indices[i] = 0;
	}

	// Ensure all threads have finished initializing shared memory
	//__syncthreads();

	// Search for the hash in the given range
	for (auto i = index; i < length; i += blockDim.x * gridDim.x) {
		if (hashes[i] == searched_hash) {
			// Atomically update the shared index array
			auto position = atomicAdd(ptr, 1);
			shared_indices[position % FIND_HASH_SHARED_MEM] = i;
		}
	}

	// Ensure all threads have finished updating shared memory
	//__syncthreads();

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
		Algorithm::EncryptionScheme scheme = Algorithm::decode(index);

		// Convert EncryptionScheme to a std::uint64_t value for hashing
		std::uint64_t combined_scheme = Algorithm::encode(scheme);

		// Hash value for combined scheme
		shared_hashes[threadIdx.x] = Hash::hash(combined_scheme);

		// Ensure all threads have finished calculating hashes
		//__syncthreads();

		// Copy shared hashes to global memory
		hashes[index] = shared_hashes[threadIdx.x];
	}
}

