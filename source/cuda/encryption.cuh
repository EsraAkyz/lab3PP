#pragma once

#include "common.cuh"

#include <cstdint>

__global__ void hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length);

__global__ void flat_hash(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length);

__global__ void find_hash(const std::uint64_t* const hashes, unsigned int* const indices, const unsigned int length, const std::uint64_t searched_hash, unsigned int* const ptr);

__global__ void hash_schemes(std::uint64_t* const hashes, const unsigned int length);
