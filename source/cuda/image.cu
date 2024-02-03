#include "image.cuh"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// task 2 b)
__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height) {


	// input[0] bis input[width*height -1]
for (auto y = std::uint32_t(0); y < (width*height); y++) {                  // -1? 
	const auto pixel = input[y];
	const auto r = pixel.get_red_channel();
	const auto g = pixel.get_green_channel();
	const auto b = pixel.get_blue_channel();

	const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
	const auto gray_converted = static_cast<std::uint8_t>(gray);

			const auto gray_pixel = BitmapImage::BitmapPixel{ gray_converted , gray_converted,  gray_converted };

			output[y] = gray_pixel;
	
	}
}

// task 2 c)
BitmapImage get_grayscale_cuda(const BitmapImage& source) {
	BitmapImage bitmapImage(source.get_height(), source.get_width());
	int index = 0; 
	
	Pixel<std::uint8_t>* const output = nullptr;                              // unsicher
	const BitmapImage::BitmapPixel* input = source.get_data();

	grayscale_kernel(input,output,source.get_width(), source.get_height());

	for (auto i = 0; i < bitmapImage.get_height(); i++) {
		for (auto j = 0; j < bitmapImage.get_width(); j++) {
			bitmapImage.set_pixel(i,j,output[index]);                         // i j Reihenfolge korrekt?
			index++;
		}
	}
	return bitmapImage;
}


