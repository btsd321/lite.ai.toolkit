//
// GpuMaskDecoder 实现
//

#include "gpu_mask_decode.h"
#include <cstring>
#include <iostream>

namespace trtgpu
{

GpuMaskDecoder::GpuMaskDecoder(int proto_c, float mask_threshold)
    : proto_c_(proto_c), mask_threshold_(mask_threshold)
{
}

std::vector<cv::cuda::GpuMat> GpuMaskDecoder::decode(
    const float* proto_device,
    int proto_h, int proto_w,
    const std::vector<DetectionInfo>& detections,
    int input_h, int input_w,
    int pad_left, int pad_top,
    int resize_w, int resize_h,
    int img_h, int img_w,
    cudaStream_t stream)
{
    int num_det = static_cast<int>(detections.size());
    if (num_det <= 0) return {};

    size_t mask_area = static_cast<size_t>(img_h) * img_w;

    // 分配 GPU 缓冲
    coeffs_buf_.reserve(num_det * proto_c_ * sizeof(float));
    bboxes_buf_.reserve(num_det * 4 * sizeof(float));
    masks_buf_.reserve(num_det * mask_area);

    // 打包 coeffs 和 bboxes 到连续数组
    std::vector<float> coeffs_host(num_det * proto_c_);
    std::vector<float> bboxes_host(num_det * 4);

    for (int i = 0; i < num_det; ++i)
    {
        const auto& det = detections[i];
        if (static_cast<int>(det.mask_coeffs.size()) >= proto_c_)
        {
            std::memcpy(&coeffs_host[i * proto_c_],
                        det.mask_coeffs.data(),
                        proto_c_ * sizeof(float));
        }
        bboxes_host[i * 4 + 0] = det.x1;
        bboxes_host[i * 4 + 1] = det.y1;
        bboxes_host[i * 4 + 2] = det.x2;
        bboxes_host[i * 4 + 3] = det.y2;
    }

    // 上传到 GPU
    cudaMemcpyAsync(coeffs_buf_.data(), coeffs_host.data(),
                    coeffs_host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(bboxes_buf_.data(), bboxes_host.data(),
                    bboxes_host.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 清零输出缓冲（bbox 外区域保持 0）
    cudaMemsetAsync(masks_buf_.data(), 0, num_det * mask_area, stream);

    // 计算坐标映射参数：原图坐标 → proto 空间坐标
    // 原图(x,y) → 裁剪 letterbox: cx = x * resize_w / img_w
    // → letterbox: lx = cx + pad_left
    // → proto: px = lx * proto_w / input_w
    MaskDecodeParams params{};
    params.proto_h = proto_h;
    params.proto_w = proto_w;
    params.proto_c = proto_c_;
    params.img_h = img_h;
    params.img_w = img_w;
    params.scale_x = static_cast<float>(resize_w) / img_w * proto_w / input_w;
    params.scale_y = static_cast<float>(resize_h) / img_h * proto_h / input_h;
    params.offset_x = static_cast<float>(pad_left) * proto_w / input_w;
    params.offset_y = static_cast<float>(pad_top) * proto_h / input_h;
    params.mask_threshold = mask_threshold_;
    params.num_detections = num_det;

    // 启动 kernel
    launch_batch_mask_decode(
        proto_device,
        coeffs_buf_.as<const float>(),
        bboxes_buf_.as<const float>(),
        masks_buf_.as<uint8_t>(),
        params, stream);

    // 同步后创建独立 GpuMat（clone 保证独立内存所有权）
    cudaStreamSynchronize(stream);

    std::vector<cv::cuda::GpuMat> result;
    result.reserve(num_det);
    for (int i = 0; i < num_det; ++i)
    {
        uint8_t* mask_ptr = masks_buf_.as<uint8_t>() + i * mask_area;
        cv::cuda::GpuMat view(img_h, img_w, CV_8UC1, mask_ptr,
                              static_cast<size_t>(img_w));
        result.push_back(view.clone());
    }
    return result;
}

}  // namespace trtgpu
