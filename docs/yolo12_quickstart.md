# YOLO12 快速开始指南

## 快速上手

本指南帮助您快速在 lite.ai.toolkit 中使用 YOLO12 目标检测模型。

### 1. 环境准备

确保已安装必要的依赖：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake libopencv-dev

# 或者使用 conda 环境
conda install opencv cmake
```

### 2. 获取预训练模型

从 Ultralytics 官方获取 YOLO12 预训练模型：

```python
# 安装 ultralytics
pip install ultralytics

# 下载并导出模型
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo12n.pt')  # 会自动下载

# 导出 ONNX 格式
model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
```

### 3. 编译 lite.ai.toolkit

```bash
git clone https://github.com/DefTruth/lite.ai.toolkit.git
cd lite.ai.toolkit

mkdir build && cd build
cmake -DENABLE_ONNXRUNTIME=ON -DENABLE_TEST=ON ..
make -j$(nproc)
make install
```

### 4. 运行示例

```bash
# 复制模型文件到正确位置
cp yolo12n.onnx examples/hub/onnx/cv/

# 运行测试
cd build
./bin/test_lite_yolo12
```

### 5. 自定义检测

创建您自己的检测程序：

```cpp
#include "lite/lite.h"
#include <opencv2/opencv.hpp>

int main() {
    // 初始化模型
    std::string model_path = "yolo12n.onnx";
    auto detector = new lite::cv::detection::YOLO12(model_path);
    
    // 加载图片
    cv::Mat image = cv::imread("your_image.jpg");
    
    // 执行检测
    std::vector<lite::types::Boxf> boxes;
    detector->detect(image, boxes, 0.25f, 0.45f);
    
    // 绘制结果
    lite::utils::draw_boxes_inplace(image, boxes);
    
    // 保存结果
    cv::imwrite("result.jpg", image);
    
    // 输出检测信息
    std::cout << "检测到 " << boxes.size() << " 个目标" << std::endl;
    for (const auto& box : boxes) {
        std::cout << "类别: " << box.label_text 
                  << " 置信度: " << box.score
                  << " 位置: [" << box.x1 << "," << box.y1 
                  << "," << box.x2 << "," << box.y2 << "]" << std::endl;
    }
    
    delete detector;
    return 0;
}
```

### 6. CMakeLists.txt 配置

如果要在自己的项目中使用：

```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo12_demo)

set(CMAKE_CXX_STANDARD 17)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 lite.ai.toolkit
find_package(lite.ai.toolkit REQUIRED)

# 创建可执行文件
add_executable(yolo12_demo main.cpp)

# 链接库
target_link_libraries(yolo12_demo 
    ${OpenCV_LIBS}
    lite.ai.toolkit
)
```

## 常见问题

### Q: 编译时出现找不到头文件错误

A: 确保正确设置了 include 路径：
```bash
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

### Q: 运行时出现模型加载错误

A: 检查：
1. 模型文件路径是否正确
2. 模型文件是否完整
3. ONNXRuntime 版本是否兼容

### Q: 检测结果不准确

A: 尝试调整参数：
```cpp
detector->detect(image, boxes, 
                0.25f,  // score_threshold: 降低以检测更多目标
                0.45f,  // iou_threshold: NMS 阈值
                100);   // topk: 最大检测数量
```

### Q: 性能不佳

A: 优化建议：
1. 使用较小的模型（yolo12n vs yolo12x）
2. 降低输入图像分辨率
3. 使用 GPU 加速（如果支持）

## 进阶使用

### 批量处理

```cpp
// 处理多张图片
std::vector<std::string> image_paths = {"img1.jpg", "img2.jpg", "img3.jpg"};
for (const auto& path : image_paths) {
    cv::Mat img = cv::imread(path);
    std::vector<lite::types::Boxf> boxes;
    detector->detect(img, boxes);
    // 处理结果...
}
```

### 视频处理

```cpp
cv::VideoCapture cap("video.mp4");
cv::Mat frame;
while (cap.read(frame)) {
    std::vector<lite::types::Boxf> boxes;
    detector->detect(frame, boxes);
    lite::utils::draw_boxes_inplace(frame, boxes);
    cv::imshow("YOLO12 Detection", frame);
    if (cv::waitKey(1) == 27) break; // ESC 退出
}
```

### 实时摄像头检测

```cpp
cv::VideoCapture cap(0); // 使用默认摄像头
if (!cap.isOpened()) {
    std::cerr << "无法打开摄像头" << std::endl;
    return -1;
}

cv::Mat frame;
while (true) {
    cap >> frame;
    if (frame.empty()) break;
    
    std::vector<lite::types::Boxf> boxes;
    detector->detect(frame, boxes);
    lite::utils::draw_boxes_inplace(frame, boxes);
    
    cv::imshow("实时检测", frame);
    if (cv::waitKey(1) == 27) break;
}
```

## 性能基准

不同模型在 CPU 上的推理速度（仅供参考）：

| 模型 | 输入尺寸 | CPU时间 (ms) | 内存占用 (MB) |
|------|----------|--------------|---------------|
| YOLO12n | 640x640 | ~50 | ~80 |
| YOLO12s | 640x640 | ~120 | ~150 |
| YOLO12m | 640x640 | ~300 | ~280 |
| YOLO12l | 640x640 | ~450 | ~350 |
| YOLO12x | 640x640 | ~800 | ~650 |

*测试环境：Intel i7-10700K, 无GPU加速*

## 获取帮助

如果遇到问题，可以：

1. 查看 [完整文档](./yolo12_support.md)
2. 提交 [GitHub Issue](https://github.com/DefTruth/lite.ai.toolkit/issues)
3. 参考现有的 YOLO 实现示例

祝您使用愉快！🚀