//
// Created for YOLO26-OBB support
//

#include "lite/lite.h"

static void test_default()
{
    std::string onnx_path = "../../../test/model/yolo26m-obb/weights/best.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_yolo26_obb.jpg";
    std::string save_img_path = "../../../examples/logs/test_lite_yolo26_obb_default.jpg";

    // 1. Test Default Engine ONNXRuntime
    lite::cv::detection::YOLO26OBB *detector = new lite::cv::detection::YOLO26OBB(onnx_path);
    
    // Set custom class names (optional)
    std::vector<std::string> class_names = {"ExpressBillSeg", "BarCode", "2DCode"};
    detector->set_class_names(class_names);

    std::vector<lite::types::BoxfWithAngle> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    
    if (img_bgr.empty())
    {
        std::cout << "Error: Cannot read image from " << test_img_path << std::endl;
        delete detector;
        return;
    }

    detector->detect(img_bgr, detected_boxes, 0.25f, 0.45f, 300);

    // Print results
    std::cout << "Default Version Detected Boxes Num: " << detected_boxes.size() << std::endl;
    for (size_t i = 0; i < detected_boxes.size(); ++i)
    {
        const auto &box = detected_boxes[i];
        std::cout << "  [" << i << "] " << box.label_text 
                  << " | Score: " << box.score
                  << " | Angle: " << (box.angle * 180.0 / CV_PI) << "°"
                  << " | Center: (" << box.cx << ", " << box.cy << ")"
                  << " | Size: " << box.width << "x" << box.height << std::endl;
    }

    // Draw rotated boxes
    lite::utils::draw_boxes_with_angle_inplace(img_bgr, detected_boxes);
    cv::imwrite(save_img_path, img_bgr);
    
    std::cout << "Result saved to: " << save_img_path << std::endl;

    delete detector;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
    std::string onnx_path = "../../../test/model/yolo26m-obb/weights/best.onnx";
    std::string test_img_path = "../../../examples/lite/resources/test_yolo26_obb.jpg";
    std::string save_img_path = "../../../examples/logs/test_onnx_yolo26_obb.jpg";

    // 1. Test ONNXRuntime Engine
    lite::onnxruntime::cv::detection::YOLO26OBB *detector =
        new lite::onnxruntime::cv::detection::YOLO26OBB(onnx_path);

    // Set custom class names
    std::vector<std::string> class_names = {"ExpressBillSeg", "BarCode", "2DCode"};
    detector->set_class_names(class_names);

    std::vector<lite::types::BoxfWithAngle> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    
    if (img_bgr.empty())
    {
        std::cout << "Error: Cannot read image from " << test_img_path << std::endl;
        delete detector;
        return;
    }

    detector->detect(img_bgr, detected_boxes, 0.25f, 0.45f, 300);

    // Print results
    std::cout << "\n=== ONNXRuntime Version Results ===" << std::endl;
    std::cout << "Detected Boxes: " << detected_boxes.size() << std::endl;
    
    for (size_t i = 0; i < std::min(detected_boxes.size(), size_t(10)); ++i)
    {
        const auto &box = detected_boxes[i];
        std::cout << "  [" << i << "] " << box.label_text 
                  << " | Score: " << std::fixed << std::setprecision(3) << box.score
                  << " | Angle: " << std::setprecision(1) << (box.angle * 180.0 / CV_PI) << "°"
                  << " | Box: [" << std::setprecision(0) 
                  << box.x1 << ", " << box.y1 << ", " 
                  << box.x2 << ", " << box.y2 << "]" << std::endl;
    }

    // Draw rotated boxes
    lite::utils::draw_boxes_with_angle_inplace(img_bgr, detected_boxes);
    cv::imwrite(save_img_path, img_bgr);
    
    std::cout << "Result saved to: " << save_img_path << std::endl;

    delete detector;
#else
    std::cout << "ONNXRuntime is not enabled!" << std::endl;
#endif
}

static void test_tensorrt()
{
#ifdef ENABLE_TENSORRT
    std::string engine_path = "../../../test/model/yolo26m-obb/weights/best.engine";
    std::string test_img_path = "../../../examples/lite/resources/test_yolo26_obb.jpg";
    std::string save_img_path = "../../../examples/logs/test_trt_yolo26_obb.jpg";

    // 1. Test TensorRT Engine
    lite::trt::cv::detection::YOLO26OBB *detector =
        new lite::trt::cv::detection::YOLO26OBB(engine_path);

    // Set input format (OpenCV uses BGR by default)
    detector->setInputFormat(lite::trt::cv::detection::YOLO26OBB::ImageFormat::BGR);

    // Set custom class names
    std::vector<std::string> class_names = {"ExpressBillSeg", "BarCode", "2DCode"};
    detector->set_class_names(class_names);

    std::vector<lite::types::BoxfWithAngle> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    
    if (img_bgr.empty())
    {
        std::cout << "Error: Cannot read image from " << test_img_path << std::endl;
        delete detector;
        return;
    }

    detector->detect(img_bgr, detected_boxes, 0.25f, 0.45f, 300);

    // Print results
    std::cout << "\n=== TensorRT Version Results ===" << std::endl;
    std::cout << "Detected Boxes: " << detected_boxes.size() << std::endl;
    
    for (size_t i = 0; i < std::min(detected_boxes.size(), size_t(10)); ++i)
    {
        const auto &box = detected_boxes[i];
        std::cout << "  [" << i << "] " << box.label_text 
                  << " | Score: " << std::fixed << std::setprecision(3) << box.score
                  << " | Angle: " << std::setprecision(1) << (box.angle * 180.0 / CV_PI) << "°"
                  << " | Center: (" << std::setprecision(1) 
                  << box.cx << ", " << box.cy << ")"
                  << " | Size: " << box.width << "x" << box.height << std::endl;
    }

    // Draw rotated boxes
    lite::utils::draw_boxes_with_angle_inplace(img_bgr, detected_boxes);
    cv::imwrite(save_img_path, img_bgr);
    
    std::cout << "Result saved to: " << save_img_path << std::endl;

    delete detector;
#else
    std::cout << "TensorRT is not enabled!" << std::endl;
#endif
}

static void print_usage()
{
    std::cout << "\n==================================" << std::endl;
    std::cout << "YOLO26-OBB Test Utility" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "\nUsage:" << std::endl;
    std::cout << "  test_lite_yolo26_obb [backend]" << std::endl;
    std::cout << "\nBackends:" << std::endl;
    std::cout << "  default    - Default backend (ONNXRuntime)" << std::endl;
    std::cout << "  onnx       - ONNXRuntime backend" << std::endl;
    std::cout << "  trt        - TensorRT backend (GPU)" << std::endl;
    std::cout << "  all        - Run all available backends" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  test_lite_yolo26_obb" << std::endl;
    std::cout << "  test_lite_yolo26_obb onnx" << std::endl;
    std::cout << "  test_lite_yolo26_obb trt" << std::endl;
    std::cout << "  test_lite_yolo26_obb all" << std::endl;
    std::cout << "==================================" << std::endl;
}

int main(int argc, char *argv[])
{
    print_usage();
    
    std::string backend = "default";
    if (argc > 1)
    {
        backend = argv[1];
    }

    std::cout << "\nRunning test with backend: " << backend << "\n" << std::endl;

    if (backend == "default")
    {
        test_default();
    }
    else if (backend == "onnx")
    {
        test_onnxruntime();
    }
    else if (backend == "trt")
    {
        test_tensorrt();
    }
    else if (backend == "all")
    {
        std::cout << "\n>>> Testing Default Backend <<<" << std::endl;
        test_default();
        
        std::cout << "\n>>> Testing ONNXRuntime Backend <<<" << std::endl;
        test_onnxruntime();
        
        std::cout << "\n>>> Testing TensorRT Backend <<<" << std::endl;
        test_tensorrt();
    }
    else
    {
        std::cout << "Unknown backend: " << backend << std::endl;
        print_usage();
        return 1;
    }

    std::cout << "\n=== Test Completed ===" << std::endl;
    return 0;
}
