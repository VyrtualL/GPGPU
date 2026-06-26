#include "hysteresis.hpp"
#include "utils.hpp"

DeviceHysteresis::DeviceHysteresis(ImageView<rgb8> initial, int low, int high)
    : Hysteresis(initial, low, high, true)
{}

__global__ static void
initialize_intermediate_(ImageView<float> distances, const int low,
                         const int high, ImageView<uint32_t> intermediate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= intermediate.width || y < 0 || y >= intermediate.height)
        return;
    auto value = distances.get(x, y);
    if (value < low)
        intermediate.set(x, y, MASK_OFF);
    else if (value > high)
        intermediate.set(x, y, MASK_ON);
    else
        intermediate.set(x, y, MASK_CANDIDATE);
}

__device__ static int
diffuse_intermediate_value_(ImageView<uint32_t> intermediate, int x, int y)
{
    for (int dy = -1; dy <= 1; ++dy)
    {
        if (y + dy < 0 || y + dy >= intermediate.height)
            continue;
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (x + dx < 0 || x + dx >= intermediate.width)
                continue;
            if (intermediate.get(x + dx, y + dy) == MASK_ON)
                return MASK_ON;
        }
    }
    return MASK_OFF;
}

__global__ static void diffuse_intermediate_(ImageView<uint32_t> mask,
                                             ImageView<uint32_t> intermediate)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= intermediate.width || y < 0 || y >= intermediate.height)
        return;
    int value = intermediate.get(x, y);
    if (value == MASK_CANDIDATE)
        value = diffuse_intermediate_value_(intermediate, x, y);
    mask.set(x, y, value);
}

ImageView<uint32_t> DeviceHysteresis::operator()(ImageView<float> distances)
{
    assert(distances.width == this->intermediate_.width
           && distances.height == this->intermediate_.height);
    auto params = KernelParams(distances, dim3(16, 16));
    initialize_intermediate_<<<params.grid, params.block>>>(
        distances, this->low_, this->high_, this->intermediate_);
    assert(cudaGetLastError() == cudaSuccess);
    auto mask = ImageView<uint32_t>(distances);
    diffuse_intermediate_<<<params.grid, params.block>>>(mask,
                                                         this->intermediate_);
    assert(cudaGetLastError() == cudaSuccess);
    return mask;
}
