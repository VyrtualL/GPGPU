#pragma once

#include "Image.hpp"

static constexpr uint32_t MASK_OFF = 0;
static constexpr uint32_t MASK_ON = 1;
static constexpr uint32_t MASK_CANDIDATE = 2;

class Hysteresis
{
public:
    Hysteresis(ImageView<rgb8> initial, int low, int high, bool device);

    virtual ImageView<uint32_t> operator()(ImageView<float> distances) = 0;

protected:
    const int low_;
    const int high_;

    Image<uint32_t> intermediate_;
};

class HostHysteresis : public Hysteresis
{
public:
    HostHysteresis(ImageView<rgb8> initial, int low, int high);

    virtual ImageView<uint32_t> operator()(ImageView<float> distances) override;

private:
    void initialize_intermediate_(ImageView<float> distances);
    void diffuse_intermediate_(ImageView<uint32_t> mask);
    uint32_t diffuse_intermediate_value_(int x, int y) const;
};

class DeviceHysteresis : public Hysteresis
{
public:
    DeviceHysteresis(ImageView<rgb8> initial, int low, int high);

    virtual ImageView<uint32_t> operator()(ImageView<float> distances) override;

private:
};
