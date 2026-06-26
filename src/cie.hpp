#pragma once

#include "Image.hpp"

namespace cie
{
    struct XYZ
    {
        float X, Y, Z;
    };

    struct Lab
    {
        uint8_t L, a, b;
    };

    namespace host
    {
        __host__ Lab sRGB_to_Lab(rgb8 sRGB);

        __host__ float distance(Lab first, Lab second);

        __host__ Lab mean(Lab lhs, Lab rhs);
    }; // namespace host

    namespace device
    {
        __device__ Lab sRGB_to_Lab(rgb8 sRGB);

        __device__ float distance(Lab first, Lab second);

        __device__ Lab mean(Lab lhs, Lab rhs);

    } // namespace device

} // namespace cie
