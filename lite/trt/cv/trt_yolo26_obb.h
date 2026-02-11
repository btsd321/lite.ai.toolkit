//
// Created for YOLO26-OBB TensorRT support
//

#ifndef LITE_AI_TOOLKIT_TRT_YOLO26_OBB_H
#define LITE_AI_TOOLKIT_TRT_YOLO26_OBB_H

#include <algorithm>

#include "lite/trt/core/trt_core.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/utils.h"

namespace trtcv
{
class LITE_EXPORTS TRTYOLO26OBB : public BasicTRTHandler
{
public:
    enum class ImageFormat
    {
        NONE = 0,
        RGB,
        BGR
    };

    explicit TRTYOLO26OBB(const std::string &_trt_model_path, unsigned int _num_threads = 1);
    explicit TRTYOLO26OBB(const std::string &_trt_model_path, ImageFormat _input_format,
                          unsigned int _num_threads = 1);

    ~TRTYOLO26OBB() override = default;

    // 设置输入图像格式
    void setInputFormat(ImageFormat format) { input_format_ = format; }

    // 获取当前输入图像格式
    ImageFormat getInputFormat() const { return input_format_; }

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
    
    // Default COCO classes
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

    ImageFormat input_format_ = ImageFormat::RGB;  // 默认期望 RGB 格式输入

private:
    void preprocess(const cv::Mat &input_image, cv::Mat &output_mat);

    void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height, int target_width,
                        YOLO26OBBScaleParams &scale_params);

    // Detect model type based on output dimensions
    void detect_model_type();

    // For end-to-end models (with built-in NMS)
    void generate_bboxes_obb(const YOLO26OBBScaleParams &scale_params,
                             std::vector<types::BoxfWithAngle> &bbox_collection,
                             float *output,
                             float score_threshold,
                             int img_height,
                             int img_width);

    // For non-end-to-end models (without built-in NMS, like YOLOv8-OBB)
    void generate_bboxes_obb_non_end2end(const YOLO26OBBScaleParams &scale_params,
                                         std::vector<types::BoxfWithAngle> &bbox_collection,
                                         float *output,
                                         float score_threshold,
                                         int img_height,
                                         int img_width);

    void nms_obb(std::vector<types::BoxfWithAngle> &input,
                 std::vector<types::BoxfWithAngle> &output,
                 float iou_threshold,
                 unsigned int topk);

    float compute_obb_iou(const types::BoxfWithAngle &box1,
                          const types::BoxfWithAngle &box2);

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
     * @param mat Input image (BGR format by default)
     * @param detected_boxes Output vector of detected oriented boxes
     * @param score_threshold Minimum confidence threshold (default: 0.25)
     * @param iou_threshold IoU threshold (Note: NMS is already applied in TensorRT engine)
     * @param topk Maximum number of detections to return (default: 300)
     */
    void detect(const cv::Mat &mat,
                std::vector<types::BoxfWithAngle> &detected_boxes,
                float score_threshold = 0.25f,
                float iou_threshold = 0.45f,
                unsigned int topk = 300);
};
}  // namespace trtcv

#endif  // LITE_AI_TOOLKIT_TRT_YOLO26_OBB_H
