//
// GPU Buffer 实现
//

#include "gpu_buffer.h"
#include <iostream>
#include <utility>

namespace trtgpu
{

GpuBuffer::~GpuBuffer()
{
    release();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : ptr_(other.ptr_), capacity_(other.capacity_)
{
    other.ptr_ = nullptr;
    other.capacity_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept
{
    if (this != &other)
    {
        release();
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

void GpuBuffer::reserve(size_t size)
{
    if (size <= capacity_) return;
    release();
    cudaError_t err = cudaMalloc(&ptr_, size);
    if (err != cudaSuccess)
    {
        std::cerr << "GpuBuffer::reserve cudaMalloc failed (" << size
                  << " bytes): " << cudaGetErrorString(err) << std::endl;
        ptr_ = nullptr;
        capacity_ = 0;
        return;
    }
    capacity_ = size;
}

void GpuBuffer::release()
{
    if (ptr_)
    {
        cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
    }
}

}  // namespace trtgpu
