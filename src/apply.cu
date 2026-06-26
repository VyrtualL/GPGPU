#include "apply.hpp"
#include "hysteresis.hpp"
#include "utils.hpp"

DeviceApplyMask::DeviceApplyMask()
    : ApplyMask::ApplyMask()
{}

DeviceApplyMask::DeviceApplyMask(rgb8 color, float alpha)
    : ApplyMask(color, alpha)
{}

__global__ static void apply_(ImageView<rgb8> frame, ImageView<uint32_t> mask,
                              rgb8 color, float alpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= frame.width || y < 0 || y >= frame.height)
        return;
    if (mask.get(x, y) == MASK_OFF)
        return;
    frame.set(x, y, rgb8::blend(frame.get(x, y), color, alpha));
}

void DeviceApplyMask::operator()(ImageView<rgb8> frame,
                                 ImageView<uint32_t> mask)
{
    assert(frame.width == mask.width && frame.height == mask.height);
    auto params = KernelParams(frame, dim3(16, 16));
    apply_<<<params.grid, params.block>>>(frame, mask, this->color_,
                                          this->alpha_);
    assert(cudaGetLastError() == cudaSuccess);
}
