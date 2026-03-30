/**
 * YOLO26-OBB TensorRT Engine 检测 Demo
 *
 * 用法:
 *   ./yolo26_obb_demo --model <engine_path> --image <image_path> [options]
 *
 * 选项:
 *   --model         TensorRT engine 文件路径 (必填)
 *   --image         输入图片路径 (必填)
 *   --output        输出图片路径 (默认: result.jpg)
 *   --score         置信度阈值 (默认: 0.25)
 *   --iou           NMS IoU 阈值 (默认: 0.45)
 *   --topk          最大检测数量 (默认: 300)
 */

#include <iostream>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>

#include "lite/lite.h"

// 类别颜色表 (BGR)
static const cv::Scalar CLASS_COLORS[] = {
    cv::Scalar(0,   200,  0),    // 0: ExpressBillSeg — 绿
    cv::Scalar(255, 100,  0),    // 1: BarCode        — 蓝橙
    cv::Scalar(0,   0,   255),   // 2: 2DCode         — 红
};
static const int NUM_COLORS = 3;

/**
 * 在图像上绘制 OBB 旋转框及标签
 */
static void draw_obb(cv::Mat &img, const std::vector<lite::types::BoxfWithAngle> &boxes)
{
    for (const auto &box : boxes)
    {
        if (!box.flag) continue;

        // 旋转矩形 (angle 为弧度, cv::RotatedRect 接受角度制)
        float angle_deg = box.angle * 180.0f / static_cast<float>(CV_PI);
        cv::RotatedRect rrect(
            cv::Point2f(box.cx, box.cy),
            cv::Size2f(box.width, box.height),
            angle_deg
        );

        cv::Point2f pts[4];
        rrect.points(pts);

        cv::Scalar color = CLASS_COLORS[box.label % NUM_COLORS];

        // 绘制四条边
        for (int i = 0; i < 4; ++i)
            cv::line(img, pts[i], pts[(i + 1) % 4], color, 2, cv::LINE_AA);

        // 绘制标签: "ClassName 0.92"
        std::string label_str;
        if (box.label_text != nullptr)
            label_str = std::string(box.label_text);
        else
            label_str = "cls" + std::to_string(box.label);
        label_str += " " + std::to_string(box.score).substr(0, 4);

        // 标签背景
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label_str, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        cv::Point text_origin = cv::Point(
            static_cast<int>(pts[1].x),
            static_cast<int>(pts[1].y) - 4);
        // 防止越界
        text_origin.x = std::max(0, text_origin.x);
        text_origin.y = std::max(text_size.height, text_origin.y);

        cv::rectangle(img,
            text_origin + cv::Point(0, baseline),
            text_origin + cv::Point(text_size.width, -text_size.height),
            color, cv::FILLED);
        cv::putText(img, label_str, text_origin,
            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

int main(int argc, char **argv)
{
    // ---- 参数解析 ----
    argparse::ArgumentParser program("yolo26_obb_demo");

    program.add_argument("--model")
        .required()
        .help("TensorRT engine 文件路径 (.engine)");

    program.add_argument("--image")
        .required()
        .help("输入图片路径");

    program.add_argument("--output")
        .default_value(std::string("result.jpg"))
        .help("输出图片路径 (默认: result.jpg)");

    program.add_argument("--score")
        .default_value(0.25f)
        .scan<'f', float>()
        .help("置信度阈值 (默认: 0.25)");

    program.add_argument("--iou")
        .default_value(0.45f)
        .scan<'f', float>()
        .help("NMS IoU 阈值 (默认: 0.45)");

    program.add_argument("--topk")
        .default_value(300)
        .scan<'i', int>()
        .help("最大检测数量 (默认: 300)");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        std::cerr << program;
        return 1;
    }

    const std::string model_path  = program.get<std::string>("--model");
    const std::string image_path  = program.get<std::string>("--image");
    const std::string output_path = program.get<std::string>("--output");
    const float score_threshold   = program.get<float>("--score");
    const float iou_threshold     = program.get<float>("--iou");
    const int   topk              = program.get<int>("--topk");

    // ---- 加载图片 ----
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "[ERROR] 无法读取图片: " << image_path << "\n";
        return 1;
    }
    std::cout << "[INFO] 图片大小: " << img.cols << "x" << img.rows << "\n";

    // ---- 创建检测器 ----
    std::cout << "[INFO] 加载模型: " << model_path << "\n";
    auto *detector = new lite::trt::cv::detection::YOLO26OBB(model_path);

    // OpenCV 读取的图像为 BGR 格式
    detector->setInputFormat(lite::trt::cv::detection::YOLO26OBB::ImageFormat::BGR);

    // 自定义类别名称
    detector->set_class_names({"ExpressBillSeg", "BarCode", "2DCode"});

    // ---- 执行检测 ----
    std::vector<lite::types::BoxfWithAngle> boxes;
    detector->detect(img, boxes,
                     score_threshold,
                     iou_threshold,
                     static_cast<unsigned int>(topk));

    std::cout << "[INFO] 检测到 " << boxes.size() << " 个目标\n";

    // ---- 打印检测结果 ----
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        const auto &b = boxes[i];
        if (!b.flag) continue;
        float angle_deg = b.angle * 180.0f / static_cast<float>(CV_PI);
        std::cout << "  [" << i << "] "
                  << (b.label_text ? b.label_text : ("cls" + std::to_string(b.label)))
                  << "  score=" << b.score
                  << "  cx=" << b.cx << " cy=" << b.cy
                  << " w=" << b.width << " h=" << b.height
                  << " angle=" << angle_deg << "°\n";
    }

    // ---- 绘制并保存 ----
    draw_obb(img, boxes);
    cv::imwrite(output_path, img);
    std::cout << "[INFO] 结果已保存到: " << output_path << "\n";

    delete detector;
    return 0;
}
