#pragma once

#include <cassert>
#include <cstring>
#include <memory>
#include <string_view>

#include "cuda_runtime.h"
#include "stb_image.h"

struct rgb8
{
    uint8_t r, g, b;

    __device__ __host__ static rgb8 blend(rgb8 background, rgb8 foreground,
                                          float alpha) // New!
    {
        uint8_t r = (1.0F - alpha) * background.r + alpha * foreground.r;
        uint8_t g = (1.0F - alpha) * background.g + alpha * foreground.g;
        uint8_t b = (1.0F - alpha) * background.b + alpha * foreground.b;
        return { r, g, b };
    }
};

// View over a 2D buffer
template <class T>
struct ImageView
{
    T* buffer = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride = 0;

    __device__ __host__ T get(int x, int y) const; // New!
    __device__ __host__ void set(int x, int y, T value); // New!

    template <class U>
    operator ImageView<U>() const // New!
    {
        assert(sizeof(T) == sizeof(U));
        auto view = ImageView<U>();
        view.buffer = reinterpret_cast<U*>(this->buffer);
        view.width = this->width;
        view.height = this->height;
        view.stride = this->stride;
        return view;
    }
};

template <class T>
T ImageView<T>::get(int x, int y) const // New!
{
    assert(x >= 0 && x < this->width && y >= 0 && y < this->height);
    const auto* bytes = reinterpret_cast<const char*>(this->buffer);
    const auto* line = reinterpret_cast<const T*>(bytes + y * this->stride);
    return line[x];
}

template <class T>
void ImageView<T>::set(int x, int y, T value) // New!
{
    assert(x >= 0 && x < this->width && y >= 0 && y < this->height);
    auto* bytes = reinterpret_cast<char*>(this->buffer);
    auto* line = reinterpret_cast<T*>(bytes + y * this->stride);
    line[x] = value;
}

// Class that owns data
template <class T>
struct Image : ImageView<T>
{
    void (*deleter)(void*) = nullptr;

    Image() = default;
    Image(int width, int height, bool device = false);
    Image(const char* path);
    Image(std::string_view path)
        : Image(path.data())
    {}
    ~Image();

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    Image(const Image& other) = delete;
    Image& operator=(const Image& other) = delete;

    Image clone() const;
};

template <class T>
Image<T>::Image(const char* path)
{
    int w, h, n;
    this->buffer = (T*)stbi_load(path, &w, &h, &n, sizeof(T));
    this->width = w;
    this->height = h;
    this->stride = w * sizeof(T);
    this->deleter = stbi_image_free;
}

template <class T>
Image<T>::Image(int width, int height, bool device)
{
    static auto cudaDelete = [](void* ptr) { cudaFree(ptr); };
    this->width = width;
    this->height = height;
    if (device)
    {
        size_t pitch;
        cudaMallocPitch((void**)&this->buffer, &pitch, this->width * sizeof(T),
                        this->height);
        this->stride = pitch;
        this->deleter = cudaDelete;
    }
    else
    {
        this->stride = width * sizeof(T);
        this->buffer = (T*)malloc(this->height * this->stride);
        this->deleter = free;
    }
}

template <class T>
Image<T>::~Image()
{
    if (this->buffer && this->deleter)
        this->deleter(this->buffer);
    this->buffer = nullptr;
}

template <class T>
Image<T>::Image(Image&& other) noexcept
{
    std::swap((ImageView<T>&)(*this), (ImageView<T>&)(other));
}

template <class T>
Image<T>& Image<T>::operator=(Image&& other) noexcept
{
    std::swap((ImageView<T>&)(*this), (ImageView<T>&)(other));
    return *this;
}

template <class T>
Image<T> Image<T>::clone() const
{
    Image<T> out(this->width, this->height);
    for (int y = 0; y < this->height; ++y)
        std::memcpy((char*)out.buffer + y * out.stride, //
                    (char*)this->buffer + y * this->stride, //
                    this->width * sizeof(T));
    return out;
}
