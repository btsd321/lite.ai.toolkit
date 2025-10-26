//
// Created by DefTruth on 2024/10/26.
//

#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../../../examples/hub/onnx/cv/yolo12n.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_yolo12_1.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::detection::YOLO12 *yolo12 = new lite::cv::detection::YOLO12(onnx_path); // default

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "../../../examples/hub/onnx/cv/yolo12n.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_onnx_yolo12_1.jpg";

    // 1. Test ONNXRuntime Engine
    lite::onnxruntime::cv::detection::YOLO12 *yolo12 =
        new lite::onnxruntime::cv::detection::YOLO12(onnx_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "ONNXRuntime Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
    std::string mnn_path = "../../../examples/hub/mnn/cv/yolo12n.mnn";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_mnn_yolo12_1.jpg";

    // 1. Test MNN Engine
    lite::mnn::cv::detection::YOLO12 *yolo12 =
        new lite::mnn::cv::detection::YOLO12(mnn_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "MNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
    std::string param_path = "../../../examples/hub/ncnn/cv/yolo12n.param";
    std::string bin_path = "../../../examples/hub/ncnn/cv/yolo12n.bin";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_ncnn_yolo12_1.jpg";

    // 1. Test NCNN Engine
    lite::ncnn::cv::detection::YOLO12 *yolo12 =
        new lite::ncnn::cv::detection::YOLO12(param_path, bin_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "NCNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
    std::string proto_path = "../../../examples/hub/tnn/cv/yolo12n.tnnproto";
    std::string model_path = "../../../examples/hub/tnn/cv/yolo12n.tnnmodel";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_tnn_yolo12_1.jpg";

    // 1. Test TNN Engine
    lite::tnn::cv::detection::YOLO12 *yolo12 =
        new lite::tnn::cv::detection::YOLO12(proto_path, model_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "TNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../examples/hub/trt/yolo12n.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_lite_yolov5_1.jpg";
    std::string save_img_path = "../../../examples/logs/test_trt_yolo12_1.jpg";

    // Test TensorRT Engine
    lite::trt::cv::detection::YOLO12 *yolo12 =
        new lite::trt::cv::detection::YOLO12(engine_path);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    yolo12->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);

    cv::imwrite(save_img_path, img_bgr);

    std::cout << "TensorRT Version Detected Boxes Num: " << detected_boxes.size() << std::endl;

    delete yolo12;
#endif
}

static void test_lite()
{
    test_default();
    test_onnxruntime();
    test_mnn();
    test_ncnn();
    test_tnn();
    test_tensorrt();
}

int main(__unused int argc, __unused char *argv[])
{
    test_lite();
    return 0;
}