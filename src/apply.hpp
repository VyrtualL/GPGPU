#pragma once

#include "Image.hpp"

class ApplyMask
{
public:
    ApplyMask();
    ApplyMask(rgb8 color, float alpha);

    virtual void operator()(ImageView<rgb8> frame,
                            ImageView<uint32_t> mask) = 0;

protected:
    const rgb8 color_;
    const float alpha_;
};

class HostApplyMask : public ApplyMask
{
public:
    HostApplyMask();
    HostApplyMask(rgb8 color, float alpha);

    virtual void operator()(ImageView<rgb8> frame,
                            ImageView<uint32_t> mask) override;

private:
};

class DeviceApplyMask : public ApplyMask
{
public:
    DeviceApplyMask();
    DeviceApplyMask(rgb8 color, float alpha);

    virtual void operator()(ImageView<rgb8> frame,
                            ImageView<uint32_t> mask) override;

private:
};
