#include "opening.hpp"

MorphologicalOpening::MorphologicalOpening(ImageView<rgb8> initial,
                                           int kernel_size, bool device)
    : kernel_size_(kernel_size)
    , intermediate_(initial.width, initial.height, device)
{
    assert(kernel_size % 2 != 0);
}

HostMorphologicalOpening::HostMorphologicalOpening(ImageView<rgb8> initial,
                                                   int kernel_size)
    : MorphologicalOpening(initial, kernel_size, false)
{}

void HostMorphologicalOpening::operator()(ImageView<float> distances)
{
    this->erosion_(distances);
    this->dilatation_(distances);
}

void HostMorphologicalOpening::erosion_(ImageView<float> in)
{
    for (int y = 0; y < in.height; ++y)
    {
        for (int x = 0; x < in.width; ++x)
        {
            auto boundaries = this->compute_kernel_boundaries_(in, x, y);
            auto min = in.get(x, y);
            for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
                 ++kernel_y)
                for (int kernel_x = boundaries.x.min;
                     kernel_x < boundaries.x.max; ++kernel_x)
                    min = std::min(min, in.get(kernel_x, kernel_y));
            this->intermediate_.set(x, y, min);
        }
    }
}

void HostMorphologicalOpening::dilatation_(ImageView<float> in)
{
    for (int y = 0; y < this->intermediate_.height; ++y)
    {
        for (int x = 0; x < this->intermediate_.width; ++x)
        {
            auto boundaries =
                this->compute_kernel_boundaries_(this->intermediate_, x, y);
            auto max = this->intermediate_.get(x, y);
            for (int kernel_y = boundaries.y.min; kernel_y < boundaries.y.max;
                 ++kernel_y)
                for (int kernel_x = boundaries.x.min;
                     kernel_x < boundaries.x.max; ++kernel_x)
                    max = std::max(max,
                                   this->intermediate_.get(kernel_x, kernel_y));
            in.set(x, y, max);
        }
    }
}
