#include "encryption.cuh"
#include "util/Hash.h"

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
