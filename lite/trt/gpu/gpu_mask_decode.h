//
// GpuMaskDecoder — GPU 批量 Mask 解码器
// 将 TRT proto 输出（GPU）+ NMS 后的 coeffs → GPU 上的二值 mask
//

#ifndef LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_H
#define LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_H

#include "gpu_buffer.h"
#include "gpu_mask_decode.cuh"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

namespace trtgpu
{

/// 单个检测框的 NMS 后信息（CPU 端中转）
struct DetectionInfo
{
    float x1, y1, x2, y2;           // 原图坐标 bbox
    float score;
    unsigned int label;
    std::vector<float> mask_coeffs;  // 32 维 mask 系数
};

/// GPU 批量 Mask 解码器
class GpuMaskDecoder
{
public:
    /// @param proto_c proto 通道数（默认 32）
    /// @param mask_threshold 二值化阈值（默认 0.5）
    explicit GpuMaskDecoder(int proto_c = 32, float mask_threshold = 0.5f);

    ~GpuMaskDecoder() = default;

    /// 批量解码 mask（proto 零拷贝，输出独立 GpuMat）
    /// @param proto_device  TRT 输出 proto 的设备指针（直接用 buffers[2]，零拷贝）
    /// @param proto_h, proto_w proto 空间尺寸
    /// @param detections NMS 后的检测结果
    /// @param input_h, input_w 模型输入尺寸
    /// @param pad_left, pad_top letterbox padding
    /// @param resize_w, resize_h letterbox 内区域尺寸
    /// @param img_h, img_w 原始图像尺寸
    /// @param stream CUDA stream
    /// @return 每个检测框的 GPU mask（CV_8UC1, img_h × img_w, 0/255）
    std::vector<cv::cuda::GpuMat> decode(
        const float* proto_device,
        int proto_h, int proto_w,
        const std::vector<DetectionInfo>& detections,
        int input_h, int input_w,
        int pad_left, int pad_top,
        int resize_w, int resize_h,
        int img_h, int img_w,
        cudaStream_t stream);

private:
    int proto_c_;
    float mask_threshold_;

    GpuBuffer coeffs_buf_;  // [N, proto_c] float
    GpuBuffer bboxes_buf_;  // [N, 4] float
    GpuBuffer masks_buf_;   // [N, img_h, img_w] uint8
};

}  // namespace trtgpu

#endif  // LITE_AI_TOOLKIT_TRT_GPU_MASK_DECODE_H
