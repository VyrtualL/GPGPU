#include "Compute.hpp"

#include <chrono>
#include <thread>

#include "apply.hpp"
#include "background.hpp"
#include "hysteresis.hpp"
#include "opening.hpp"

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);

/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
    static auto estimator = HostBackgroundEstimator(in);
    auto distances = estimator(in);

    static auto opening = HostMorphologicalOpening(in, 3);
    opening(distances);

    static auto hysteresis = HostHysteresis(in, 3, 30);
    auto mask = hysteresis(distances);

    static auto apply_mask = HostApplyMask();
    apply_mask(in, mask);
}

extern "C"
{
    static Parameters g_params;

    void cpt_init(Parameters* params)
    {
        g_params = *params;
    }

    void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
    {
        auto img = ImageView<rgb8>{ (rgb8*)buffer, width, height, stride };
        if (g_params.device == e_device_t::CPU)
            compute_cpp(img);
        else if (g_params.device == e_device_t::GPU)
            compute_cu(img);
    }
}
