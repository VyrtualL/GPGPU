#include "hysteresis.hpp"

Hysteresis::Hysteresis(ImageView<rgb8> initial, int low, int high, bool device)
    : low_(low)
    , high_(high)
    , intermediate_(initial.width, initial.height, device)
{
    assert(low < high);
}

HostHysteresis::HostHysteresis(ImageView<rgb8> initial, int low, int high)
    : Hysteresis(initial, low, high, false)
{}

ImageView<uint32_t> HostHysteresis::operator()(ImageView<float> distances)
{
    assert(distances.width == this->intermediate_.width
           && distances.height == this->intermediate_.height);
    this->initialize_intermediate_(distances);
    auto mask = ImageView<uint32_t>(distances);
    this->diffuse_intermediate_(mask);
    return mask;
}

void HostHysteresis::initialize_intermediate_(ImageView<float> distances)
{
    for (int y = 0; y < distances.height; ++y)
    {
        for (int x = 0; x < distances.width; ++x)
        {
            auto value = distances.get(x, y);
            if (value < this->low_)
                this->intermediate_.set(x, y, MASK_OFF);
            else if (value > this->high_)
                this->intermediate_.set(x, y, MASK_ON);
            else
                this->intermediate_.set(x, y, MASK_CANDIDATE);
        }
    }
}

void HostHysteresis::diffuse_intermediate_(ImageView<uint32_t> mask)
{
    for (int y = 0; y < this->intermediate_.height; ++y)
    {
        for (int x = 0; x < this->intermediate_.width; ++x)
        {
            int value = this->intermediate_.get(x, y);
            if (value == MASK_CANDIDATE)
                value = this->diffuse_intermediate_value_(x, y);
            mask.set(x, y, value);
        }
    }
}

uint32_t HostHysteresis::diffuse_intermediate_value_(int x, int y) const
{
    for (int dy = -1; dy <= 1; ++dy)
    {
        if (y + dy < 0 || y + dy >= this->intermediate_.height)
            continue;
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (x + dx < 0 || x + dx >= this->intermediate_.width)
                continue;
            if (this->intermediate_.get(x + dx, y + dy) == MASK_ON)
                return MASK_ON;
        }
    }
    return MASK_OFF;
}
