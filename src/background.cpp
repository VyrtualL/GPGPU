#include "background.hpp"

#include <cassert>
#include <cmath>

#include "cie.hpp"

BackgroundEstimator::BackgroundEstimator(ImageView<rgb8> initial, bool device)
    : background_(initial.width, initial.height, device)
    , candidate_(initial.width, initial.height, device)
    , time_(initial.width, initial.height, device)
    , distances_(initial.width, initial.height, device)
{}

HostBackgroundEstimator::HostBackgroundEstimator(ImageView<rgb8> initial)
    : BackgroundEstimator(initial, false)
{
    for (int y = 0; y < initial.height; ++y)
        for (int x = 0; x < initial.width; ++x)
            this->background_.set(x, y,
                                  cie::host::sRGB_to_Lab(initial.get(x, y)));
    cudaMemcpy2D(this->candidate_.buffer, this->candidate_.stride,
                 this->background_.buffer, this->background_.stride,
                 this->background_.width * sizeof(*this->background_.buffer),
                 this->background_.height, cudaMemcpyHostToHost);
    assert(cudaGetLastError() == cudaSuccess);
    memset(this->time_.buffer, 0, this->time_.height * this->time_.stride);
}

ImageView<float> HostBackgroundEstimator::operator()(ImageView<rgb8> frame)
{
    assert(frame.width == this->background_.width
           && frame.height == this->background_.height);
    for (int y = 0; y < frame.height; ++y)
        for (int x = 0; x < frame.width; ++x)
            this->step_(frame, x, y);
    return this->distances_;
}

void HostBackgroundEstimator::step_(ImageView<rgb8> frame, const int x,
                                    const int y)
{
    const auto background = this->background_.get(x, y);
    const auto current = cie::host::sRGB_to_Lab(frame.get(x, y));
    const auto candidate = this->candidate_.get(x, y);
    const auto distance = cie::host::distance(background, current);
    if (distance >= 25)
    {
        auto time = this->time_.get(x, y);
        if (time == 0)
        {
            this->candidate_.set(x, y, current);
            this->time_.set(x, y, time + 1);
        }
        else if (time < 100)
        {
            this->candidate_.set(x, y, cie::host::mean(candidate, current));
            this->time_.set(x, y, time + 1);
        }
        else
        {
            this->background_.set(x, y, candidate);
            this->candidate_.set(x, y, background);
            this->time_.set(x, y, 0);
        }
    }
    else
    {
        this->background_.set(x, y, cie::host::mean(background, current));
        this->time_.set(x, y, 0);
    }
    this->distances_.set(x, y, distance);
}
