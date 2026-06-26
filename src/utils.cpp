#include "utils.hpp"

#include <cmath>

KernelParams::KernelParams(int width, int height, dim3 block_dim)
{
    assert(width > 0 && height > 0);

    int device_count;
    cudaGetDeviceCount(&device_count);
    assert(cudaGetLastError() == cudaSuccess);
    assert(device_count > 0);

    cudaDeviceProp props;
    int device_target = 0;
    cudaGetDeviceProperties(&props, device_target);
    assert(cudaGetLastError() == cudaSuccess);
    assert(!(props.major == 9999 && props.minor == 9999));

    this->block = block_dim;
    assert(static_cast<int>(this->block.x) <= props.maxThreadsDim[0]
           && static_cast<int>(this->block.y) <= props.maxThreadsDim[1]);

    this->grid = dim3(std::ceil(static_cast<float>(width) / this->block.x),
                      std::ceil(static_cast<float>(height) / this->block.y));
    assert(static_cast<int>(this->grid.x) <= props.maxGridSize[0]
           && static_cast<int>(this->grid.y) <= props.maxGridSize[1]);
}
