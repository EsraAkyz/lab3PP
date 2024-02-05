#pragma once

#ifndef __global__
// Falls der NVCC die Datei inkludiert, hat dieser __global__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __global__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __global__
#endif

#ifndef __device__
// Falls der NVCC die Datei inkludiert, hat dieser __device__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __device__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __device__
#endif

#ifndef __host__
// Falls der NVCC die Datei inkludiert, hat dieser __host__ bereits definiert und ignoriert das hier.
// Falls der Host-Compiler die Datei inkludiert, ist dieser von __host__ verwirrt, also sagen wir ihm, das ist eigentlich leer.
#define __host__
#endif

#if defined (__INTELLISENSE__) | defined (__RESHARPER__)
// Hier können diverse Funktionsdeklarationen eingefügt werden, die der NVCC kennt, der Host-Compiler aber nicht.
// Wenn die in CUDA-Kerneln aufgerufen werden, zeigt sonst die IDE einen Fehler an.
// Hier beispielsweise mit atomicAdd, vergleiche https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-functions
template<class T1>
__device__ T1 atomicAdd(T1* x, T1 y);
#endif

#define GRAYSCALE_SHARED_MEM 0
#define HASH_SHARED_MEM 256
#define FLAT_HASH_SHARED_MEM 256
#define FIND_HASH_SHARED_MEM 256
#define HASH_SCHEMES_SHARED_MEM 256

// task 2 a)

inline [[nodiscard]] unsigned int divup(unsigned int numThreads, unsigned int threadsPerBlock){      
	return (numThreads+ threadsPerBlock -1) / threadsPerBlock;
}
