//
// YOLOv8-Seg instance segmentation example
//

#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path      = "../../../examples/hub/onnx/cv/yolov8n-seg.onnx";
    std::string test_img_path  = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path  = "../../../examples/logs/test_lite_yolov8seg_1.jpg";

    lite::cv::instance_segmentation::YOLOv8Seg *segmentor =
        new lite::cv::instance_segmentation::YOLOv8Seg(onnx_path);

    std::vector<lite::types::BoxfWithSegMask> detected_objects;
    cv::Mat img_bgr = cv::imread(test_img_path);
    segmentor->detect(img_bgr, detected_objects);

    lite::utils::draw_seg_masks_inplace(img_bgr, detected_objects);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "Default (ONNXRuntime) detected objects: " << detected_objects.size() << std::endl;
    delete segmentor;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path      = "../../../examples/hub/onnx/cv/yolov8n-seg.onnx";
    std::string test_img_path  = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path  = "../../../examples/logs/test_onnx_yolov8seg_1.jpg";

    lite::onnxruntime::cv::instance_segmentation::YOLOv8Seg *segmentor =
        new lite::onnxruntime::cv::instance_segmentation::YOLOv8Seg(onnx_path);

    std::vector<lite::types::BoxfWithSegMask> detected_objects;
    cv::Mat img_bgr = cv::imread(test_img_path);
    segmentor->detect(img_bgr, detected_objects);

    lite::utils::draw_seg_masks_inplace(img_bgr, detected_objects);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "ONNXRuntime detected objects: " << detected_objects.size() << std::endl;
    delete segmentor;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path    = "../../../examples/hub/trt/yolov8n-seg_fp32.engine";
    std::string test_img_path  = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path  = "../../../examples/logs/test_trt_yolov8seg_1.jpg";

    lite::trt::cv::instance_segmentation::YOLOV8Seg *segmentor =
        new lite::trt::cv::instance_segmentation::YOLOV8Seg(engine_path);

    std::vector<lite::types::BoxfWithSegMask> detected_objects;
    cv::Mat img_bgr = cv::imread(test_img_path);
    segmentor->detect(img_bgr, detected_objects);

    lite::utils::draw_seg_masks_inplace(img_bgr, detected_objects);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "TensorRT detected objects: " << detected_objects.size() << std::endl;
    delete segmentor;
#endif
}

static void test_lite()
{
    test_default();
    test_onnxruntime();
    test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}
