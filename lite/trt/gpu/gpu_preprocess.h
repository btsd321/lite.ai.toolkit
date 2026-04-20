//
// GpuPreprocessor — GPU 融合预处理器
// 将输入图像（CPU 或 GPU）直接预处理到 TRT 输入 buffer
//

#ifndef LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_H
#define LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_H

#include "gpu_buffer.h"
#include "gpu_preprocess.cuh"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace trtgpu
{

/// Letterbox 缩放参数（CPU 端，用于后处理坐标反算）
struct ScaleParams
{
    float ratio = 1.f;    // 缩放比例 min(dst/src)
    int pad_left = 0;     // 左侧 padding 像素数
    int pad_top = 0;      // 顶部 padding 像素数
    int resized_w = 0;    // 缩放后宽度（不含 padding）
    int resized_h = 0;    // 缩放后高度（不含 padding）
    int src_w = 0;        // 原始图像宽度
    int src_h = 0;        // 原始图像高度
};

/// GPU 融合预处理器
class GpuPreprocessor
{
public:
    /// @param model_h 模型输入高度
    /// @param model_w 模型输入宽度
    /// @param pad_value letterbox 填充像素值（uint8，默认 114）
    GpuPreprocessor(int model_h, int model_w, float pad_value = 114.f);

    ~GpuPreprocessor() = default;

    /// 从 CPU cv::Mat 预处理，结果直接写入 trt_input_buffer
    void preprocess(const cv::Mat& cpu_image, float* trt_input_buffer,
                    ScaleParams& scale_params, cudaStream_t stream,
                    bool bgr2rgb = true);

    /// 从 GPU GpuMat 预处理（零拷贝输入）
    void preprocess(const cv::cuda::GpuMat& gpu_image, float* trt_input_buffer,
                    ScaleParams& scale_params, cudaStream_t stream,
                    bool bgr2rgb = true);

private:
    void compute_params(int src_h, int src_w, bool bgr2rgb,
                        ScaleParams& scale_params,
                        LetterboxParams& kernel_params) const;

    int model_h_;
    int model_w_;
    float pad_value_norm_;  // pad_value / 255.0
    GpuBuffer upload_buf_;  // CPU→GPU upload 缓冲
};

}  // namespace trtgpu

#endif  // LITE_AI_TOOLKIT_TRT_GPU_PREPROCESS_H
