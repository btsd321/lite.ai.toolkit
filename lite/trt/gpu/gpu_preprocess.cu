//
// Fused GPU preprocess CUDA kernel 实现
//

#include "gpu_preprocess.cuh"

namespace trtgpu
{

__global__ void fused_preprocess_kernel(
    const uint8_t* __restrict__ src,
    int src_pitch,
    float* __restrict__ dst,
    LetterboxParams p)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= p.dst_w || dy >= p.dst_h) return;

    float c0, c1, c2;  // 三通道插值结果

    // 在 letterbox 内的相对坐标
    int rx = dx - p.pad_left;
    int ry = dy - p.pad_top;

    if (rx < 0 || rx >= p.resize_w || ry < 0 || ry >= p.resize_h)
    {
        // padding 区域
        c0 = c1 = c2 = p.pad_value;
    }
    else
    {
        // 映射到源图坐标（OpenCV INTER_LINEAR 的 half-pixel center 对齐）
        float sx = (rx + 0.5f) * p.src_w / p.resize_w - 0.5f;
        float sy = (ry + 0.5f) * p.src_h / p.resize_h - 0.5f;

        // 钳位到 [0, src_size - 1]
        sx = fmaxf(0.f, fminf(sx, (float)(p.src_w - 1)));
        sy = fmaxf(0.f, fminf(sy, (float)(p.src_h - 1)));

        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = min(x0 + 1, p.src_w - 1);
        int y1 = min(y0 + 1, p.src_h - 1);

        float fx = sx - x0;
        float fy = sy - y0;
        float w00 = (1.f - fx) * (1.f - fy);
        float w10 = fx * (1.f - fy);
        float w01 = (1.f - fx) * fy;
        float w11 = fx * fy;

        // 采样 4 个邻域像素（3 通道 HWC 布局）
        const uint8_t* row0 = src + y0 * src_pitch;
        const uint8_t* row1 = src + y1 * src_pitch;

        c0 = (w00 * row0[x0 * 3 + 0] + w10 * row0[x1 * 3 + 0] +
              w01 * row1[x0 * 3 + 0] + w11 * row1[x1 * 3 + 0]) * p.norm_factor;
        c1 = (w00 * row0[x0 * 3 + 1] + w10 * row0[x1 * 3 + 1] +
              w01 * row1[x0 * 3 + 1] + w11 * row1[x1 * 3 + 1]) * p.norm_factor;
        c2 = (w00 * row0[x0 * 3 + 2] + w10 * row0[x1 * 3 + 2] +
              w01 * row1[x0 * 3 + 2] + w11 * row1[x1 * 3 + 2]) * p.norm_factor;
    }

    // 写入 CHW 布局，同时处理 BGR→RGB
    int area = p.dst_h * p.dst_w;
    int offset = dy * p.dst_w + dx;
    if (p.bgr2rgb)
    {
        dst[0 * area + offset] = c2;  // R ← B
        dst[1 * area + offset] = c1;  // G ← G
        dst[2 * area + offset] = c0;  // B ← R
    }
    else
    {
        dst[0 * area + offset] = c0;
        dst[1 * area + offset] = c1;
        dst[2 * area + offset] = c2;
    }
}

void launch_fused_preprocess(
    const uint8_t* src,
    int src_pitch,
    float* dst,
    const LetterboxParams& params,
    cudaStream_t stream)
{
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid((params.dst_w + block.x - 1) / block.x,
              (params.dst_h + block.y - 1) / block.y);

    fused_preprocess_kernel<<<grid, block, 0, stream>>>(src, src_pitch, dst, params);
}

}  // namespace trtgpu
