#include <iostream>

#include "background.hpp"
#include "utils.hpp"

__global__ static void initialize_background_(ImageView<cie::Lab> background,
                                              ImageView<rgb8> initial)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= initial.width || y < 0 || y >= initial.height)
        return;
    background.set(x, y, cie::device::sRGB_to_Lab(initial.get(x, y)));
}

__global__ static void step_(ImageView<cie::Lab> background,
                             ImageView<cie::Lab> candidate,
                             ImageView<uint32_t> time,
                             ImageView<float> distances, ImageView<rgb8> frame)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= background.width || y < 0 || y >= background.height)
        return;
    const auto val_background = background.get(x, y);
    const auto val_current = cie::device::sRGB_to_Lab(frame.get(x, y));
    const auto val_candidate = candidate.get(x, y);
    const auto distance = cie::device::distance(val_background, val_current);
    if (distance >= 25)
    {
        auto val_time = time.get(x, y);
        if (val_time == 0)
        {
            candidate.set(x, y, val_current);
            time.set(x, y, val_time + 1);
        }
        else if (val_time < 100)
        {
            candidate.set(x, y, cie::device::mean(val_candidate, val_current));
            time.set(x, y, val_time + 1);
        }
        else
        {
            background.set(x, y, val_candidate);
            candidate.set(x, y, val_background);
            time.set(x, y, 0);
        }
    }
    else
    {
        background.set(x, y, cie::device::mean(val_background, val_current));
        time.set(x, y, 0);
    }
    distances.set(x, y, distance);
}

DeviceBackgroundEstimator::DeviceBackgroundEstimator(ImageView<rgb8> initial)
    : BackgroundEstimator(initial, true)
{
    auto params = KernelParams(initial, dim3(16, 16));
    initialize_background_<<<params.grid, params.block>>>(this->background_,
                                                          initial);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy2D(this->candidate_.buffer, this->candidate_.stride,
                 this->background_.buffer, this->background_.stride,
                 this->background_.width * sizeof(*this->background_.buffer),
                 this->background_.height, cudaMemcpyDeviceToDevice);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset2D(this->time_.buffer, this->time_.stride, 0,
                 this->time_.width * sizeof(*this->time_.buffer),
                 this->time_.height);
    assert(cudaGetLastError() == cudaSuccess);
}

ImageView<float> DeviceBackgroundEstimator::operator()(ImageView<rgb8> frame)
{
    assert(frame.width == this->background_.width
           && frame.height == this->background_.height);
    auto params = KernelParams(frame, dim3(16, 16));
    step_<<<params.grid, params.block>>>(this->background_, this->candidate_,
                                         this->time_, this->distances_, frame);
    assert(cudaGetLastError() == cudaSuccess);
    return this->distances_;
}
