//
// Created for YOLOv8-Seg (instance segmentation) support
//

#ifndef LITE_AI_ORT_CV_YOLOV8SEG_H
#define LITE_AI_ORT_CV_YOLOV8SEG_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
    class LITE_EXPORTS YOLOv8Seg : public BasicOrtHandler
    {
    public:
        explicit YOLOv8Seg(const std::string &_onnx_path, unsigned int _num_threads = 1)
            : BasicOrtHandler(_onnx_path, _num_threads) {};

        ~YOLOv8Seg() override = default;

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
        } YOLOv8SegScaleParams;

    private:
        static constexpr const float mean_val = 0.f;
        static constexpr const float scale_val = 1.0 / 255.f;
        static constexpr const float mask_threshold = 0.5f;
        static constexpr const int num_mask_coeffs = 32; // prototype mask channels

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
        Ort::Value transform(const cv::Mat &mat_rs) override;

        void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                            int target_height, int target_width,
                            YOLOv8SegScaleParams &scale_params);

        // Decode boxes and collect mask coefficients (before NMS)
        // detection_results: each entry is {Boxf, vector<float> mask_coeffs(32)}
        void generate_detections(
            const YOLOv8SegScaleParams &scale_params,
            std::vector<types::Boxf> &bbox_collection,
            std::vector<std::vector<float>> &mask_coeffs_collection,
            std::vector<Ort::Value> &output_tensors,
            float score_threshold, int img_height, int img_width);

        void nms(std::vector<types::Boxf> &input,
                 std::vector<types::Boxf> &output,
                 std::vector<std::vector<float>> &coeffs_in,
                 std::vector<std::vector<float>> &coeffs_out,
                 float iou_threshold, unsigned int topk, unsigned int nms_type);

        // Decode a single instance mask from prototypes + coefficients
        cv::Mat decode_single_mask(
            const std::vector<float> &coeffs,
            const float *proto_data,
            int proto_c, int proto_h, int proto_w,
            const YOLOv8SegScaleParams &scale_params,
            int img_h, int img_w,
            float x1, float y1, float x2, float y2);

    public:
        void detect(const cv::Mat &mat,
                    std::vector<types::BoxfWithSegMask> &detected_objects,
                    float score_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    unsigned int topk = 100,
                    unsigned int nms_type = NMS::OFFSET);
    };
} // namespace ortcv

#endif // LITE_AI_ORT_CV_YOLOV8SEG_H
