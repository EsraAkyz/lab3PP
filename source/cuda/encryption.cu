#include "encryption.cuh"
#include "util/Hash.h"
#include "util/SharedMemory.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/** Kernel of hash function **/
__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){
    // Calculate the global index of the current thread
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within length
    if(index < length){
        std::uint64_t hash_value = Hash::hash(values[index]);
        hashes[index] = hash_value;
    }
}

/** Kernel of flat_hash **/
#define FLAT_HASH_SHARED_MEM 64 // 128 or 64???
__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length){
    // Allocate shared memory
    __shared__ std::uint64_t shared_values[FLAT_HASH_SHARED_MEM_SIZE];
    __shared__ std::uint64_t shared_hashes[FLAT_HASH_SHARED_MEM_SIZE];

    // Calculate the global index of current thread
    unsigned int index = threadIdx.x; // Cause the kernel is invoked with (1, 1, 1) thread blocks of size (tx, 1, 1)

    // Check if thread within length
    if(index < length){
        // Load the value into shared memory
        shared_values[index] = values[index];
        __syncthreads();  // Ensure all threads have loaded their values

        // Calculate the hash value for the corresponding value
        shared_hashes[index] = Hash::hash(shared_values[index]);
        __syncthreads();  // Ensure all threads have calculated their hashes

        // Write the hash value to global memory
        hashes[index] = shared_hashes[index];
    }
}


/** Kernel for find_hash **/
#define FIND_HASH_SHARED_MEM_SIZE 64
__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr){
    // Calculate the global index of the current thread
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory declaration
    __shared__ unsigned int shared_indices[FIND_HASH_SHARED_MEM_SIZE];

    // Initialize shared memory
    for (unsigned int i = threadIdx.x; i < FIND_HASH_SHARED_MEM_SIZE; i += blockDim.x) {
        shared_indices[i] = 0;
    }

    // Ensure all threads have finished initializing shared memory
    __syncthreads();

    // Search for the hash in the given range
    for (unsigned int i = index; i < length; i += blockDim.x * gridDim.x) {
        if (hashes[i] == searched_hash) {
            // Atomically update the shared index array
            unsigned int position = atomicAdd(ptr, 1);
            shared_indices[position % FIND_HASH_SHARED_MEM_SIZE] = i;
        }
    }

    // Ensure all threads have finished updating shared memory
    __syncthreads();

    // Copy shared indices to global memory
    for (unsigned int i = threadIdx.x; i < FIND_HASH_SHARED_MEM_SIZE; i += blockDim.x) {
        if (shared_indices[i] > 0) {
            unsigned int position = atomicAdd(ptr, 1);
            indices[position] = shared_indices[i];
        }
    }
}
