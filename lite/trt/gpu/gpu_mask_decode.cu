//
// 批量 GPU mask decode CUDA kernel 实现
//

#include "gpu_mask_decode.cuh"

namespace trtgpu
{

/// 每个 block 处理一个检测框的部分像素
/// grid: (ceil(img_w/32), ceil(img_h/8), num_detections)
/// block: (32, 8, 1)
/// 使用 shared memory 缓存 coeffs（每个检测 32 floats = 128 bytes）
__global__ void batch_mask_decode_kernel(
    const float* __restrict__ proto_data,
    const float* __restrict__ coeffs_data,
    const float* __restrict__ bboxes_data,
    uint8_t* __restrict__ masks_out,
    MaskDecodeParams p)
{
    int det_id = blockIdx.z;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= p.img_w || out_y >= p.img_h) return;

    // 加载当前检测框的 coeffs 到 shared memory
    __shared__ float s_coeffs[32];
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (local_tid < p.proto_c)
    {
        s_coeffs[local_tid] = coeffs_data[det_id * p.proto_c + local_tid];
    }
    __syncthreads();

    // 输出像素地址
    size_t mask_area = static_cast<size_t>(p.img_h) * p.img_w;
    uint8_t* out = masks_out + det_id * mask_area + out_y * p.img_w + out_x;

    // 检查是否在 bbox 内
    float bx1 = bboxes_data[det_id * 4 + 0];
    float by1 = bboxes_data[det_id * 4 + 1];
    float bx2 = bboxes_data[det_id * 4 + 2];
    float by2 = bboxes_data[det_id * 4 + 3];

    if (out_x < bx1 || out_x >= bx2 || out_y < by1 || out_y >= by2)
    {
        *out = 0;
        return;
    }

    // 映射到 proto 空间
    float px = out_x * p.scale_x + p.offset_x;
    float py = out_y * p.scale_y + p.offset_y;

    // 钳位
    px = fmaxf(0.f, fminf(px, (float)(p.proto_w - 1)));
    py = fmaxf(0.f, fminf(py, (float)(p.proto_h - 1)));

    // 双线性插值坐标
    int px0 = (int)floorf(px);
    int py0 = (int)floorf(py);
    int px1 = min(px0 + 1, p.proto_w - 1);
    int py1 = min(py0 + 1, p.proto_h - 1);
    float fx = px - px0;
    float fy = py - py0;
    float w00 = (1.f - fx) * (1.f - fy);
    float w10 = fx * (1.f - fy);
    float w01 = (1.f - fx) * fy;
    float w11 = fx * fy;

    // 预计算 proto 平面偏移
    int base00 = py0 * p.proto_w + px0;
    int base10 = py0 * p.proto_w + px1;
    int base01 = py1 * p.proto_w + px0;
    int base11 = py1 * p.proto_w + px1;
    int plane_size = p.proto_h * p.proto_w;

    // coeffs · proto 点乘（双线性插值采样）
    float val = 0.f;
    for (int k = 0; k < p.proto_c; ++k)
    {
        int off = k * plane_size;
        float sampled = w00 * proto_data[off + base00] +
                        w10 * proto_data[off + base10] +
                        w01 * proto_data[off + base01] +
                        w11 * proto_data[off + base11];
        val += s_coeffs[k] * sampled;
    }

    // sigmoid
    val = 1.f / (1.f + expf(-val));

    // threshold → 二值 mask
    *out = (val > p.mask_threshold) ? 255 : 0;
}

void launch_batch_mask_decode(
    const float* proto_data,
    const float* coeffs_data,
    const float* bboxes_data,
    uint8_t* masks_out,
    const MaskDecodeParams& params,
    cudaStream_t stream)
{
    if (params.num_detections <= 0) return;

    dim3 block(32, 8);  // 256 threads per block
    dim3 grid(
        (params.img_w + block.x - 1) / block.x,
        (params.img_h + block.y - 1) / block.y,
        params.num_detections);

    batch_mask_decode_kernel<<<grid, block, 0, stream>>>(
        proto_data, coeffs_data, bboxes_data, masks_out, params);
}

}  // namespace trtgpu
