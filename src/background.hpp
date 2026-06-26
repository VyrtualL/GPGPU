#pragma once

#include "cie.hpp"

class BackgroundEstimator
{
public:
    BackgroundEstimator(ImageView<rgb8> initial, bool device);

    virtual ImageView<float> operator()(ImageView<rgb8> frame) = 0;

protected:
    Image<cie::Lab> background_;
    Image<cie::Lab> candidate_;
    Image<uint32_t> time_;
    Image<float> distances_;
};

class HostBackgroundEstimator : public BackgroundEstimator
{
public:
    HostBackgroundEstimator(ImageView<rgb8> initial);

    virtual ImageView<float> operator()(ImageView<rgb8> frame) override;

private:
    void step_(ImageView<rgb8> frame, int x, int y);
};

class DeviceBackgroundEstimator : public BackgroundEstimator
{
public:
    DeviceBackgroundEstimator(ImageView<rgb8> initial);

    virtual ImageView<float> operator()(ImageView<rgb8> frame) override;

private:
};
