#include "apply.hpp"

#include "hysteresis.hpp"

ApplyMask::ApplyMask()
    : ApplyMask::ApplyMask({ 255, 0, 0 }, 0.35F)
{}

ApplyMask::ApplyMask(rgb8 color, float alpha)
    : color_(color)
    , alpha_(alpha)
{
    assert(0.0F <= alpha && alpha <= 1.0F);
}

HostApplyMask::HostApplyMask()
    : ApplyMask::ApplyMask()
{}

HostApplyMask::HostApplyMask(rgb8 color, float alpha)
    : ApplyMask(color, alpha)
{}

void HostApplyMask::operator()(ImageView<rgb8> frame, ImageView<uint32_t> mask)
{
    assert(frame.width == mask.width && frame.height == mask.height);
    for (int y = 0; y < frame.height; ++y)
    {
        for (int x = 0; x < frame.width; ++x)
        {
            if (mask.get(x, y) == MASK_OFF)
                continue;
            frame.set(x, y,
                      rgb8::blend(frame.get(x, y), this->color_, this->alpha_));
        }
    }
}
