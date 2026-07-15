//
// Created for YOLO26-Seg TensorRT support
//
// YOLO26 端到端（NMS-free）实例分割推理器。
// 与 TRTYoloV8Seg 公开 API 对齐（detect / detect_gpu / set_class_names /
// 嵌套 GpuSegDetection），便于上层通过统一接口多态调用。
//
// 与 YOLOv8-Seg 的关键差异：
//   - output0 形状为检测优先 [1, N, 4+1+1+32]（如 [1, 300, 38]），而非
//     YOLOv8 的通道优先 [1, 116, 8400]
//   - 引擎内部已完成 NMS，后处理只需按 score 阈值过滤，无需再做 NMS
//   - 每条检测的 38 个值含义：4 框 + 1 conf + 1 class_id + 32 mask 系数
//   - output1（proto，[1, 32, 160, 160]）与 YOLOv8-Seg 一致，mask 解码逻辑可复用
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLO26SEG_H
#define LITE_AI_TOOLKIT_TRT_YOLO26SEG_H

#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/trt/gpu/gpu_preprocess.h"
#include "lite/trt/gpu/gpu_mask_decode.h"
#include "lite/utils.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <opencv2/core/cuda.hpp>

namespace trtcv
{
    class LITE_EXPORTS TRTYolo26Seg : public BasicTRTHandler
    {
    public:
        explicit TRTYolo26Seg(const std::string &_trt_model_path, unsigned int _num_threads = 1)
            : BasicTRTHandler(_trt_model_path, _num_threads) {};

        ~TRTYolo26Seg() override = default;

        void set_class_names(const std::vector<std::string> &names)
        {
            custom_class_names = names;
            use_custom_class_names = true;
        }

    private:
        // Letterbox scale params
        typedef struct
        {
            float r;
            int dw;
            int dh;
            int new_unpad_w;
            int new_unpad_h;
            bool flag;
        } YOLO26SegScaleParams;

    private:
        static constexpr const float mean_val        = 0.f;
        static constexpr const float scale_val       = 1.0 / 255.f;
        static constexpr const float mask_threshold  = 0.5f;
        static constexpr const int   num_mask_coeffs = 32;
        // 每条检测的非 mask 通道数：x1,y1,x2,y2,score,class_id
        static constexpr const int   num_box_fields  = 6;

        bool use_custom_class_names = false;
        std::vector<std::string> custom_class_names;

        const char *class_names[80] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"};

    private:
        void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                            int target_height, int target_width,
                            YOLO26SegScaleParams &scale_params);

        // YOLO26 end-to-end: output0 是检测优先 (1, N, 4+1+1+32)，已 NMS，只需 score 过滤
        void generate_detections(
            const YOLO26SegScaleParams &scale_params,
            std::vector<types::Boxf> &bbox_collection,
            std::vector<std::vector<float>> &mask_coeffs_collection,
            const float *det_output,
            float score_threshold, int img_height, int img_width);

        cv::Mat decode_single_mask(
            const std::vector<float> &coeffs,
            const float *proto_data,
            int proto_c, int proto_h, int proto_w,
            const YOLO26SegScaleParams &scale_params,
            int img_h, int img_w,
            float x1, float y1, float x2, float y2);

    public:
        void detect(const cv::Mat &mat,
                    std::vector<types::BoxfWithSegMask> &detected_objects,
                    const types::InferParams &params = types::InferParams());

        /// GPU 推理结果：bbox + GPU 上的二值 mask
        struct GpuSegDetection
        {
            types::Boxf box;
            cv::cuda::GpuMat gpu_mask;  // CV_8UC1, 原图尺寸, 0/255
            bool flag = false;
        };

        /// GPU 输入 + GPU mask 输出（cv::Mat 输入会自动上传）
        void detect_gpu(const cv::Mat &mat,
                        std::vector<GpuSegDetection> &detected_objects,
                        const types::InferParams &params = types::InferParams());

        /// GPU 输入 + GPU mask 输出（零拷贝 GpuMat 输入）
        void detect_gpu(const cv::cuda::GpuMat &gpu_mat,
                        int img_height, int img_width,
                        std::vector<GpuSegDetection> &detected_objects,
                        const types::InferParams &params = types::InferParams());

    private:
        /// detect_gpu 核心实现（预处理已完成，从 TRT 推理开始）
        void detect_gpu_impl(const trtgpu::ScaleParams &scale_params,
                             int img_height, int img_width,
                             std::vector<GpuSegDetection> &detected_objects,
                             const types::InferParams &params);

        /// 按需初始化 GPU 组件
        void ensure_gpu_components();

        std::unique_ptr<trtgpu::GpuPreprocessor> gpu_preprocessor_;
        std::unique_ptr<trtgpu::GpuMaskDecoder> gpu_mask_decoder_;
    };
} // namespace trtcv

#endif // LITE_AI_TOOLKIT_TRT_YOLO26SEG_H
