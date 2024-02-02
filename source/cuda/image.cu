#include "image.cuh"
#include "common.cuh"
#include "bitmap_image.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#incude <cstdint>

/** grayscale_kernel **/
__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output,
                                 const unsigned int width, const unsigned int height){
    // Calculate the global index of the current thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the image dimensions
    if (idx < width && idy < height) {
        // Calculate the index of the current pixel
        unsigned int index = idy * width + idx;

        // Get the RGB values of the current pixel
        std::uint8_t r = input[index].get_red_channel();
        std::uint8_t g = input[index].get_green_channel();
        std::uint8_t b = input[index].get_blue_channel();

        // Calculate the grayscale value using the luminosity method
        // Y = 0.299 * R + 0.587 * G + 0.114 * B
        std::uint8_t grayscale = static_cast<std::uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);

        // Set the grayscale value to the output pixel
        output[index] = { grayscale, grayscale, grayscale };
    }
}


/** get_grayscale_cuda function **/
BitmapImage get_grayscale_cuda(const BitmapImage& source){
    // Get image dimensions
    unsigned int width = source.get_width();
    unsigned int height = source.get_height();

    // Output image
    BitmapImage result(height,width);

    // Pointers for source (input) and result (output) images
    Pixel<std::uint8_t>* d_source;
    Pixel<std::uint8_t>* d_result;

    // Allocate memory on the GPU
    cudaMalloc(&d_source, width * height * sizeof(Pixel<std::uint8_t>));
    cudaMalloc(&d_result, width * height * sizeof(Pixel<std::uint8_t>));

    // Copy source to GPU
    cudaMemcpy(d_source, source.get_data(), width * height * sizeof(Pixel<std::uint8_t>), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(32, 32);  // From 1c we use 32x32 bits (?)
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    grayscale_kernel<<<gridDim, blockDim>>>(d_source, d_result, width, height);

    // Copy the result back to the CPU
    cudaMemcpy(result.get_data(), d_result, width * height * sizeof(Pixel<std::uint8_t>), cudaMemcpyDeviceToHost);

    // Free allocated GPU memory
    cudaFree(d_source);
    cudaFree(d_result);

    return result;

}
