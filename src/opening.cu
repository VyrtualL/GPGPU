#include "opening.hpp"
#include "utils.hpp"

DeviceMorphologicalOpening::DeviceMorphologicalOpening(ImageView<rgb8> initial,
                                                       int kernel_size)
    : MorphologicalOpening(initial, kernel_size, true)
{}

__global__ static void erosion_kernel_(ImageView<float> intermediate,
                                       const int kernel_size,
                                       ImageView<float> distances)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= distances.width || y < 0 || y >= distances.height)
        return;
    const auto boundaries =
        compute_kernel_boundaries_(distances, x, y, kernel_size);
    auto min = distances.get(x, y);
    for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
         ++kernel_y)
        for (int kernel_x = boundaries.x.min; kernel_x < boundaries.x.max;
             ++kernel_x)
            min = fmin(min, distances.get(kernel_x, kernel_y));
    intermediate.set(x, y, min);
}

__global__ static void erosion_kernel_2_(ImageView<float> intermediate,
                                         const int kernel_size,
                                         ImageView<float> distances)
{
    extern __shared__ float shared_memory[];
    const int block_w = blockDim.x + kernel_size - 1;
    const int shared_x = threadIdx.x + kernel_size / 2;
    const int shared_y = threadIdx.y + kernel_size / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= distances.width || y < 0 || y >= distances.height)
        return;
    shared_memory[shared_y * block_w + shared_x] = distances.get(x, y);
    __syncthreads();
    const auto boundaries =
        compute_kernel_boundaries_(distances, x, y, kernel_size);
    auto min = distances.get(x, y);
    for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
         ++kernel_y)
    {
        for (int kernel_x = boundaries.x.min; kernel_x < boundaries.x.max;
             ++kernel_x)
        {
            min = fmin(min,
                       shared_memory[(shared_y + (kernel_y - y)) * block_w
                                     + (shared_x + (kernel_x - x))]);
        }
    }
    intermediate.set(x, y, min);
}

__global__ static void dilatation_kernel_(ImageView<float> intermediate,
                                          const int kernel_size,
                                          ImageView<float> distances)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= distances.width || y < 0 || y >= distances.height)
        return;
    const auto boundaries =
        compute_kernel_boundaries_(intermediate, x, y, kernel_size);
    auto max = intermediate.get(x, y);
    for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
         ++kernel_y)
        for (int kernel_x = boundaries.x.min; kernel_x < boundaries.x.max;
             ++kernel_x)
            max = fmax(max, intermediate.get(kernel_x, kernel_y));
    distances.set(x, y, max);
}

__global__ static void dilatation_kernel_2_(ImageView<float> intermediate,
                                            const int kernel_size,
                                            ImageView<float> distances)
{
    extern __shared__ float shared_memory[];
    const int block_w = blockDim.x + kernel_size - 1;
    const int shared_x = threadIdx.x + kernel_size / 2;
    const int shared_y = threadIdx.y + kernel_size / 2;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= distances.width || y < 0 || y >= distances.height)
        return;
    shared_memory[shared_y * block_w + shared_x] = distances.get(x, y);
    __syncthreads();
    const auto boundaries =
        compute_kernel_boundaries_(distances, x, y, kernel_size);
    auto max = distances.get(x, y);
    for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
         ++kernel_y)
    {
        for (int kernel_x = boundaries.x.min; kernel_x < boundaries.x.max;
             ++kernel_x)
        {
            max = fmax(max,
                       shared_memory[(shared_y + (kernel_y - y)) * block_w
                                     + (shared_x + (kernel_x - x))]);
        }
    }
    intermediate.set(x, y, max);
}

void DeviceMorphologicalOpening::operator()(ImageView<float> distances)
{
    auto params = KernelParams(distances, dim3(16, 16));
    size_t memory_size = (params.block.x + this->kernel_size_ - 1)
        * (params.block.y + this->kernel_size_ - 1) * sizeof(float);
    erosion_kernel_2_<<<params.grid, params.block, memory_size>>>(
        this->intermediate_, this->kernel_size_, distances);
    assert(cudaGetLastError() == cudaSuccess);
    dilatation_kernel_2_<<<params.grid, params.block, memory_size>>>(
        this->intermediate_, this->kernel_size_, distances);
    assert(cudaGetLastError() == cudaSuccess);
}
