//
// GPU Buffer — 持久化 GPU 内存管理，避免 per-frame cudaMalloc/cudaFree
//

#ifndef LITE_AI_TOOLKIT_TRT_GPU_BUFFER_H
#define LITE_AI_TOOLKIT_TRT_GPU_BUFFER_H

#include <cuda_runtime.h>
#include <cstddef>

namespace trtgpu
{

/// 持久化 GPU 内存块，自动扩容，RAII 析构释放
class GpuBuffer
{
public:
    GpuBuffer() = default;
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    /// 确保至少有 size 字节可用（容量不足时重新分配，内容不保留）
    void reserve(size_t size);

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t capacity() const { return capacity_; }

    void release();

    template <typename T>
    T* as() { return static_cast<T*>(ptr_); }

    template <typename T>
    const T* as() const { return static_cast<const T*>(ptr_); }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace trtgpu

#endif  // LITE_AI_TOOLKIT_TRT_GPU_BUFFER_H
