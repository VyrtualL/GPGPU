#include "Compute.hpp"
#include "apply.hpp"
#include "background.hpp"
#include "hysteresis.hpp"
#include "opening.hpp"

void compute_cu(ImageView<rgb8> in)
{
    static auto d_in = Image<rgb8>(in.width, in.height, true);
    cudaMemcpy2D(d_in.buffer, d_in.stride, in.buffer, in.stride,
                 in.width * sizeof(*in.buffer), in.height,
                 cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    static auto estimator = DeviceBackgroundEstimator(d_in);
    auto d_distances = estimator(d_in);

    static auto opening = DeviceMorphologicalOpening(d_in, 3);
    opening(d_distances);

    static auto hysteresis = DeviceHysteresis(d_in, 3, 30);
    auto d_mask = hysteresis(d_distances);

    static auto apply_mask = DeviceApplyMask();
    apply_mask(d_in, d_mask);

    cudaMemcpy2D(in.buffer, in.stride, d_in.buffer, d_in.stride,
                 d_in.width * sizeof(*d_in.buffer), d_in.height,
                 cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
}
