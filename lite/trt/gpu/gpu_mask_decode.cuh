//
// 批量 GPU mask decode CUDA kernel 声明
// coeffs × proto → sigmoid → resize → crop → threshold
//

#ifndef LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_CUH
#define LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace trtgpu
{

/// Mask decode kernel 参数
struct MaskDecodeParams
{
    int proto_h, proto_w;   // proto 空间尺寸（如 160×160）
    int proto_c;            // proto 通道数（如 32）
    int img_h, img_w;       // 原始图像尺寸

    // 从原图像素到 proto 空间的映射：proto_coord = pixel * scale + offset
    float scale_x, scale_y;
    float offset_x, offset_y;

    float mask_threshold;   // 二值化阈值（默认 0.5）
    int num_detections;     // 当前帧检测数量
};

/// 批量 mask decode kernel
/// @param proto_data   设备指针 [proto_c, proto_h, proto_w]（TRT output buffer，零拷贝）
/// @param coeffs_data  设备指针 [num_detections, proto_c]
/// @param bboxes_data  设备指针 [num_detections, 4]（x1, y1, x2, y2，原图坐标）
/// @param masks_out    设备指针 [num_detections, img_h, img_w]（uint8, 0/255）
/// @param params       解码参数
/// @param stream       CUDA stream
void launch_batch_mask_decode(
    const float* proto_data,
    const float* coeffs_data,
    const float* bboxes_data,
    uint8_t* masks_out,
    const MaskDecodeParams& params,
    cudaStream_t stream);

}  // namespace trtgpu

#endif  // LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_CUH
