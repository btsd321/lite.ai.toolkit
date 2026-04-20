//
// Fused GPU preprocess CUDA kernel 声明
// letterbox resize + BGR→RGB + normalize + HWC→CHW，一次 kernel 完成
//

#ifndef LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_CUH
#define LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace trtgpu
{

/// Letterbox 预处理 kernel 参数
struct LetterboxParams
{
    int src_w, src_h;       // 原始图像尺寸
    int dst_w, dst_h;       // 模型输入尺寸（如 640×640）
    int pad_left, pad_top;  // letterbox padding 偏移
    int resize_w, resize_h; // 缩放后实际区域尺寸（不含 padding）
    float norm_factor;      // 归一化因子（1/255）
    float pad_value;        // padding 填充值（已归一化，如 114/255）
    int bgr2rgb;            // 是否做 BGR→RGB 通道翻转（int 代替 bool 保证 ABI）
};

/// 融合预处理 kernel：bilinear 采样 + letterbox padding + normalize + HWC→CHW
/// @param src      设备指针：输入图像（uint8, HWC, src_h × src_w × 3）
/// @param src_pitch 输入图像的行步长（字节）
/// @param dst      设备指针：输出 tensor（float32, CHW, 3 × dst_h × dst_w）
/// @param params   预处理参数
/// @param stream   CUDA stream
void launch_fused_preprocess(
    const uint8_t* src,
    int src_pitch,
    float* dst,
    const LetterboxParams& params,
    cudaStream_t stream);

}  // namespace trtgpu

#endif  // LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_CUH
