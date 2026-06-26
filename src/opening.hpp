#pragma once

#include "Image.hpp"

struct KernelBoundaries
{
    struct
    {
        int min, max;
    } x, y;
};

class MorphologicalOpening
{
public:
    MorphologicalOpening(ImageView<rgb8> initial, int kernel_size, bool device);

    virtual void operator()(ImageView<float> distances) = 0;

protected:
    const int kernel_size_;

    Image<float> intermediate_;
};

class HostMorphologicalOpening : public MorphologicalOpening
{
public:
    HostMorphologicalOpening(ImageView<rgb8> initial, int kernel_size = 3);

    virtual void operator()(ImageView<float> distances) override;

private:
    void erosion_(ImageView<float> frame);
    void dilatation_(ImageView<float> frame);

    template <typename T>
    KernelBoundaries compute_kernel_boundaries_(ImageView<T> frame, int x,
                                                int y);
};

class DeviceMorphologicalOpening : public MorphologicalOpening
{
public:
    DeviceMorphologicalOpening(ImageView<rgb8> initial, int kernel_size = 3);

    virtual void operator()(ImageView<float> distances) override;

private:
};

template <typename T>
KernelBoundaries
HostMorphologicalOpening::compute_kernel_boundaries_(ImageView<T> frame,
                                                     const int x, const int y)
{
    auto bounds = KernelBoundaries();
    bounds.x.min = std::max(0, x - this->kernel_size_ / 2);
    bounds.x.max = std::min(frame.width, x + this->kernel_size_ / 2);
    bounds.y.min = std::max(0, y - this->kernel_size_ / 2);
    bounds.y.max = std::min(frame.height, y + this->kernel_size_ / 2);
    return bounds;
}

// std::min is not __device__.
template <typename T>
__device__ static inline T min_(T x, T y)
{
    return x < y ? x : y;
}

// std::max is not __device__.
template <typename T>
__device__ static inline T max_(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
__device__ KernelBoundaries compute_kernel_boundaries_(ImageView<T> frame,
                                                       const int x, const int y,
                                                       const int kernel_size)
{
    auto bounds = KernelBoundaries();
    bounds.x.min = max_(0, x - kernel_size / 2);
    bounds.x.max = min_(frame.width, x + kernel_size / 2);
    bounds.y.min = max_(0, y - kernel_size / 2);
    bounds.y.max = min_(frame.height, y + kernel_size / 2);
    return bounds;
}
