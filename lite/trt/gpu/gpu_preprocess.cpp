//
// GpuPreprocessor 实现
//

#include "gpu_preprocess.h"
#include <algorithm>
#include <iostream>

namespace trtgpu
{

GpuPreprocessor::GpuPreprocessor(int model_h, int model_w, float pad_value)
    : model_h_(model_h), model_w_(model_w),
      pad_value_norm_(pad_value / 255.f)
{
}

void GpuPreprocessor::compute_params(
    int src_h, int src_w, bool bgr2rgb,
    ScaleParams& scale_params,
    LetterboxParams& kernel_params) const
{
    float w_r = static_cast<float>(model_w_) / static_cast<float>(src_w);
    float h_r = static_cast<float>(model_h_) / static_cast<float>(src_h);
    float r = std::min(w_r, h_r);

    int resize_w = static_cast<int>(static_cast<float>(src_w) * r);
    int resize_h = static_cast<int>(static_cast<float>(src_h) * r);
    int pad_left = (model_w_ - resize_w) / 2;
    int pad_top = (model_h_ - resize_h) / 2;

    // CPU 端缩放参数
    scale_params.ratio = r;
    scale_params.pad_left = pad_left;
    scale_params.pad_top = pad_top;
    scale_params.resized_w = resize_w;
    scale_params.resized_h = resize_h;
    scale_params.src_w = src_w;
    scale_params.src_h = src_h;

    // GPU kernel 参数
    kernel_params.src_w = src_w;
    kernel_params.src_h = src_h;
    kernel_params.dst_w = model_w_;
    kernel_params.dst_h = model_h_;
    kernel_params.pad_left = pad_left;
    kernel_params.pad_top = pad_top;
    kernel_params.resize_w = resize_w;
    kernel_params.resize_h = resize_h;
    kernel_params.norm_factor = 1.f / 255.f;
    kernel_params.pad_value = pad_value_norm_;
    kernel_params.bgr2rgb = bgr2rgb ? 1 : 0;
}

void GpuPreprocessor::preprocess(
    const cv::Mat& cpu_image, float* trt_input_buffer,
    ScaleParams& scale_params, cudaStream_t stream,
    bool bgr2rgb)
{
    if (cpu_image.empty()) return;

    // 确保连续内存
    cv::Mat img = cpu_image.isContinuous() ? cpu_image : cpu_image.clone();

    int src_h = img.rows;
    int src_w = img.cols;
    size_t img_bytes = static_cast<size_t>(src_h) * src_w * img.channels();

    // 上传到持久化 GPU 缓冲
    upload_buf_.reserve(img_bytes);
    cudaMemcpyAsync(upload_buf_.data(), img.data, img_bytes,
                    cudaMemcpyHostToDevice, stream);

    // 计算参数
    LetterboxParams kp{};
    compute_params(src_h, src_w, bgr2rgb, scale_params, kp);

    // 启动融合预处理 kernel
    int pitch = src_w * img.channels();  // 连续内存，step = width * channels
    launch_fused_preprocess(
        upload_buf_.as<const uint8_t>(), pitch,
        trt_input_buffer, kp, stream);
}

void GpuPreprocessor::preprocess(
    const cv::cuda::GpuMat& gpu_image, float* trt_input_buffer,
    ScaleParams& scale_params, cudaStream_t stream,
    bool bgr2rgb)
{
    if (gpu_image.empty()) return;

    int src_h = gpu_image.rows;
    int src_w = gpu_image.cols;

    LetterboxParams kp{};
    compute_params(src_h, src_w, bgr2rgb, scale_params, kp);

    // 直接使用 GpuMat 的设备指针（零拷贝）
    launch_fused_preprocess(
        gpu_image.data, static_cast<int>(gpu_image.step),
        trt_input_buffer, kp, stream);
}

}  // namespace trtgpu
