#pragma once

#include "common.cuh"

#include "image/bitmap_image.h"
#include "image/pixel.h"

__global__ void grayscale_kernel(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height);

BitmapImage get_grayscale_cuda(const BitmapImage& source);
