#include "image.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "common.cuh"


// task 2 b)
__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {

	// Calculate the global index of the current thread
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if the thread is within the image dimensions
	if (idx < width && idy < height) {
		// Calculate the index of the current pixel
		unsigned int index = idy * width + idx;
                 
		const auto pixel = input[index];
		const auto r = pixel.get_red_channel();
		const auto g = pixel.get_green_channel();
		const auto b = pixel.get_blue_channel();

		const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
		const auto gray_converted = static_cast<std::uint8_t>(gray);

		const auto gray_pixel = BitmapImage::BitmapPixel{ gray_converted , gray_converted,  gray_converted };

		output[index] = gray_pixel;

	}
}

// task 2 c)
BitmapImage get_grayscale_cuda(const BitmapImage& source) {
	/*
	BitmapImage bitmapImage(source.get_height(), source.get_width());
	int index = 0;

	Pixel<std::uint8_t>* const output = nullptr;
	const BitmapImage::BitmapPixel* input = source.get_data();

	grayscale_kernel(input,output,source.get_width(), source.get_height());

	for (auto i = 0; i < bitmapImage.get_height(); i++) {
		for (auto j = 0; j < bitmapImage.get_width(); j++) {
			bitmapImage.set_pixel(i,j,output[index]);
			index++;
		}
	}
	return bitmapImage;
	*/
	// Get image dimensions
	unsigned int width = source.get_width();
	unsigned int height = source.get_height();

	// Output image
	BitmapImage result(height, width);

	// Pointers for source (input) and result (output) images
	BitmapImage::BitmapPixel* d_source = nullptr;
	BitmapImage::BitmapPixel* d_result = nullptr;

	// Allocate memory on the GPU
	cudaMalloc((void**) & d_source, width * height * sizeof(BitmapImage::BitmapPixel));
	cudaMalloc((void**) & d_result, width * height * sizeof(BitmapImage::BitmapPixel));

	// Copy source to GPU
	cudaMemcpy(d_source, source.get_data(), width * height * sizeof(Pixel<std::uint8_t>), cudaMemcpyHostToDevice);

	// Define block and grid dimensions
	dim3 blockDim(32,32);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);


	// Launch the CUDA kernel
	grayscale_kernel <<<gridDim, blockDim >>> (d_source, d_result, width, height);                                       // korrigieren  

	// Copy the result back to the CPU
	cudaMemcpy(result.get_data(), d_result, width * height * sizeof(BitmapImage::BitmapPixel), 
		cudaMemcpyDeviceToHost);

	// Free allocated GPU memory
	cudaFree(d_source);
	cudaFree(d_result);

	return result;
}

