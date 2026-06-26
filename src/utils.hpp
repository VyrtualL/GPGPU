#pragma once

#include "Image.hpp"

struct KernelParams
{
    dim3 block;
    dim3 grid;

    template <typename T>
    KernelParams(ImageView<T> view, dim3 block_dim);
    KernelParams(int width, int height, dim3 block_dim);
};

template <typename T>
KernelParams::KernelParams(ImageView<T> view, dim3 block_dim)
    : KernelParams(view.width, view.height, block_dim)
{}
