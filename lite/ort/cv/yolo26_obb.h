//
// Created for YOLO26-OBB support
//

#ifndef LITE_AI_ORT_CV_YOLO26_OBB_H
#define LITE_AI_ORT_CV_YOLO26_OBB_H

#include "lite/ort/core/ort_core.h"

namespace ortcv
{
    class LITE_EXPORTS YOLO26OBB : public BasicOrtHandler
    {
    public:
        explicit YOLO26OBB(const std::string &_onnx_path, unsigned int _num_threads = 1) 
            : BasicOrtHandler(_onnx_path, _num_threads) 
        {
            // Initialize with default class names (can be customized)
            init_default_class_names();
        };

        ~YOLO26OBB() override = default;

        // Method to set custom class names
        void set_class_names(const std::vector<std::string> &names)
        {
            custom_class_names = names;
            use_custom_class_names = true;
        }

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
        } YOLO26OBBScaleParams;

    private:
        static constexpr const float mean_val = 0.f;
        static constexpr const float scale_val = 1.0 / 255.f;
        
        // Flag to determine which class names to use
        bool use_custom_class_names = false;
        std::vector<std::string> custom_class_names;
        
        // Default COCO classes (can be overridden)
        const char *default_class_names[80] = {
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
        Ort::Value transform(const cv::Mat &mat_rs) override;

        void resize_unscale(const cv::Mat &mat,
                            cv::Mat &mat_rs,
                            int target_height,
                            int target_width,
                            YOLO26OBBScaleParams &scale_params);

        void generate_bboxes_obb(const YOLO26OBBScaleParams &scale_params,
                                 std::vector<types::BoxfWithAngle> &bbox_collection,
                                 std::vector<Ort::Value> &output_tensors,
                                 float score_threshold,
                                 int img_height,
                                 int img_width);

        void init_default_class_names()
        {
            // This is called in constructor to initialize default names
        }

        const char *get_class_name(unsigned int label)
        {
            if (use_custom_class_names && label < custom_class_names.size())
            {
                return custom_class_names[label].c_str();
            }
            else if (label < 80)
            {
                return default_class_names[label];
            }
            return "unknown";
        }

    public:
        /**
         * @brief Detect oriented bounding boxes (OBB) in an image.
         * 
         * @param mat Input image (BGR format)
         * @param detected_boxes Output vector of detected oriented boxes
         * @param score_threshold Minimum confidence threshold (default: 0.25)
         * @param iou_threshold IoU threshold for NMS (Note: NMS is already applied in ONNX model)
         * @param topk Maximum number of detections to return (default: 300)
         */
        void detect(const cv::Mat &mat, 
                    std::vector<types::BoxfWithAngle> &detected_boxes,
                    float score_threshold = 0.25f, 
                    float iou_threshold = 0.45f,
                    unsigned int topk = 300);
    };
}

#endif // LITE_AI_ORT_CV_YOLO26_OBB_H
