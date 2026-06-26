#include "cie.hpp"

#include <cmath>

namespace cie::host
{
    // Reference white D65.
    static constexpr XYZ D65 = { 0.95047F, 1.0F, 1.08883F };

    __host__ Lab sRGB_to_Lab(rgb8 sRGB)
    {
        const float nNormalizedR = sRGB.r / 255.0F;
        const float nNormalizedG = sRGB.g / 255.0F;
        const float nNormalizedB = sRGB.b / 255.0F;
        const float nX = 0.412453F * nNormalizedR + 0.35758F * nNormalizedG
            + 0.180423F * nNormalizedB;
        const float nY = 0.212671F * nNormalizedR + 0.71516F * nNormalizedG
            + 0.072169F * nNormalizedB;
        const float nZ = 0.019334F * nNormalizedR + 0.119193F * nNormalizedG
            + 0.950227F * nNormalizedB;
        float nL = cbrtf(nY);
        float nA;
        float nB;
        float nfX = nX / D65.X;
        float nfY = nY;
        float nfZ = nZ / D65.Z;
        nfY = nL - 16.0F;
        nL = 116.0F * nL - 16.0F;
        nA = cbrtf(nfX) - 16.0F;
        nA = 500.0F * (nA - nfY);
        nB = cbrtf(nfZ) - 16.0F;
        nB = 200.0F * (nfY - nB);
        uint8_t L = nL * 255.0F / 100.0F;
        uint8_t a = nA + 128.0F;
        uint8_t b = nB + 128.0F;
        return { L, a, b };
    }

    __host__ float distance(Lab first, Lab second)
    {
        const int16_t dL =
            static_cast<int16_t>(first.L) - static_cast<int16_t>(second.L);
        const int16_t da =
            static_cast<int16_t>(first.a) - static_cast<int16_t>(second.a);
        const int16_t db =
            static_cast<int16_t>(first.b) - static_cast<int16_t>(second.b);
        return std::sqrt(dL * dL + da * da + db * db);
    }

    __host__ Lab mean(Lab lhs, Lab rhs)
    {
        const uint8_t L =
            (static_cast<uint16_t>(lhs.L) + static_cast<uint16_t>(rhs.L)) / 2;
        const uint8_t a =
            (static_cast<uint16_t>(lhs.a) + static_cast<uint16_t>(rhs.a)) / 2;
        const uint8_t b =
            (static_cast<uint16_t>(lhs.b) + static_cast<uint16_t>(rhs.b)) / 2;
        return { L, a, b };
    }

} // namespace cie::host
