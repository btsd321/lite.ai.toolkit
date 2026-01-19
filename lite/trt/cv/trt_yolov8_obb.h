//
// Created for YOLOv8-OBB support
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLOV8_OBB_H
#define LITE_AI_TOOLKIT_TRT_YOLOV8_OBB_H

#include "lite/trt/core/trt_core.h"
#include "lite/utils.h"
#include "lite/trt/core/trt_utils.h"
#include <algorithm>
#include <cmath>

namespace trtcv
{
    class LITE_EXPORTS TRTYoloV8OBB : public BasicTRTHandler
    {
    public:
        explicit TRTYoloV8OBB(const std::string &_trt_model_path, unsigned int _num_threads = 1) : BasicTRTHandler(_trt_model_path, _num_threads) {};

        ~TRTYoloV8OBB() override = default;

    private:
        static constexpr const float mean_val = 0.f;
        static constexpr const float scale_val = 1.0 / 255.f;

        // Default COCO classes, can be overridden
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
        void preprocess(cv::Mat &input_image);

        void generate_bboxes_obb(
            std::vector<types::BoxfWithAngle> &bbox_collection,
            float *output,
            float score_threshold,
            int img_height,
            int img_width); // rescale & exclude with OBB angle

        void nms_obb(
            std::vector<types::BoxfWithAngle> &input,
            std::vector<types::BoxfWithAngle> &output,
            float iou_threshold,
            unsigned int topk);

        float compute_obb_iou(
            const types::BoxfWithAngle &box1,
            const types::BoxfWithAngle &box2);

    public:
        void detect(const cv::Mat &mat,
                    std::vector<types::BoxfWithAngle> &detected_boxes,
                    float score_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    unsigned int topk = 100);
    };
}

#endif // LITE_AI_TOOLKIT_TRT_YOLOV8_OBB_H
