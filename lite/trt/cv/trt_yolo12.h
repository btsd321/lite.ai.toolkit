//
// Created by DefTruth on 2024/10/26.
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLO12_H
#define LITE_AI_TOOLKIT_TRT_YOLO12_H

#include "lite/trt/core/trt_core.h"
#include "lite/utils.h"
#include "lite/trt/core/trt_utils.h"
#include <algorithm>

namespace trtcv
{
    class LITE_EXPORTS TRTYOLO12 : public BasicTRTHandler
    {
    public:
        explicit TRTYOLO12(const std::string &_trt_model_path, unsigned int _num_threads = 1) : BasicTRTHandler(_trt_model_path, _num_threads) {};

        ~TRTYOLO12() override = default;

    private:
        // nested classes
        typedef struct
        {
            float r;
            int dw;
            int dh;
            int new_unpad_w;
            int new_unpad_h;
            bool flag;
        } YOLO12ScaleParams;

    private:
        static constexpr const float mean_val = 0.f;
        static constexpr const float scale_val = 1.0 / 255.f;
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
        enum NMS
        {
            HARD = 0,
            BLEND = 1,
            OFFSET = 2
        };
        static constexpr const unsigned int max_nms = 30000;

    private:
        void preprocess(const cv::Mat &input_image, cv::Mat &output_mat);

        void resize_unscale(const cv::Mat &mat,
                            cv::Mat &mat_rs,
                            int target_height,
                            int target_width,
                            YOLO12ScaleParams &scale_params);

        void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size, YOLO12ScaleParams &scale_params);

        void generate_bboxes(const YOLO12ScaleParams &scale_params,
                             std::vector<types::Boxf> &bbox_collection,
                             float *output,
                             float score_threshold, int img_height,
                             int img_width); // rescale & exclude

        void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                 float iou_threshold, unsigned int topk, unsigned int nms_type);

    public:
        void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                    float score_threshold = 0.25f, float iou_threshold = 0.45f,
                    unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
    };
}

#endif // LITE_AI_TOOLKIT_TRT_YOLO12_H