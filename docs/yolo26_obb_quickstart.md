# YOLO26-OBB 快速入门指南

## 简介

本文档提供 YOLO26-OBB 旋转目标检测模型在 lite.ai.toolkit 中的快速入门指南。

## 功能概述

YOLO26-OBB 是一个用于旋转目标检测（Oriented Bounding Box Detection）的深度学习模型：

- ✅ 检测任意方向的目标
- ✅ 输出旋转角度信息
- ✅ 支持 ONNXRuntime 和 TensorRT 后端
- ✅ 自定义类别名称
- ✅ 端到端 NMS 后处理

## 快速开始（5分钟）

### 步骤 1: 准备模型

```bash
cd /path/to/lite.ai.toolkit/test/model

# 如果已有 PT 模型，导出 ONNX
conda activate yolo
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolo26m-obb/weights/best.pt')
model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
EOF
```

### 步骤 2: 编写代码

创建 `demo.cpp`：

```cpp
#include "lite/lite.h"

int main()
{
    // 1. 创建检测器
    auto *detector = new lite::onnxruntime::cv::detection::YOLO26OBB(
        "yolo26m-obb/weights/best.onnx");
    
    // 2. 设置类别名称（可选）
    detector->set_class_names({"ExpressBillSeg", "BarCode", "2DCode"});
    
    // 3. 加载图像
    cv::Mat img = cv::imread("test.jpg");
    
    // 4. 执行检测
    std::vector<lite::types::BoxfWithAngle> boxes;
    detector->detect(img, boxes, 0.25f);  // 置信度阈值 0.25
    
    // 5. 绘制结果
    lite::utils::draw_boxes_with_angle_inplace(img, boxes);
    cv::imwrite("result.jpg", img);
    
    // 6. 打印结果
    for (const auto &box : boxes)
    {
        std::cout << box.label_text << ": " << box.score 
                  << " @ " << (box.angle * 180.0 / CV_PI) << "°\n";
    }
    
    delete detector;
    return 0;
}
```

### 步骤 3: 编译和运行

```bash
# 编译项目
cd lite.ai.toolkit
mkdir build && cd build
cmake -DENABLE_ONNXRUNTIME=ON ..
make -j$(nproc)

# 编译你的 demo
g++ demo.cpp -o demo \
    -I../lite \
    -L./lib -llite.ai.toolkit \
    `pkg-config --cflags --libs opencv4`

# 运行
./demo
```

## 输出结果

程序会生成：

1. **图像文件**: `result.jpg` - 带旋转框的可视化结果
2. **控制台输出**: 检测框信息，包括类别、置信度、角度等

## 旋转框结构

```cpp
struct BoxfWithAngle {
    float x1, y1, x2, y2;     // 轴对齐边界框（外接矩形）
    float cx, cy;             // 旋转框中心点
    float width, height;      // 旋转框宽高
    float angle;              // 旋转角度（弧度）
    float score;              // 置信度 [0, 1]
    unsigned int label;       // 类别 ID
    const char *label_text;   // 类别名称
};
```

## TensorRT 加速（推荐用于生产环境）

### 导出 TensorRT Engine

```python
from ultralytics import YOLO

model = YOLO('yolo26m-obb/weights/best.pt')
model.export(format='engine', device=0, half=True)  # FP16 精度
```

### 使用 TensorRT

```cpp
#include "lite/lite.h"

int main()
{
    // 使用 TensorRT 后端（GPU 加速）
    auto *detector = new lite::trt::cv::detection::YOLO26OBB(
        "yolo26m-obb/weights/best.engine");
    
    // 设置输入格式（OpenCV 默认 BGR）
    detector->setInputFormat(
        lite::trt::cv::detection::YOLO26OBB::ImageFormat::BGR);
    
    // 检测（其余代码与 ONNXRuntime 相同）
    std::vector<lite::types::BoxfWithAngle> boxes;
    cv::Mat img = cv::imread("test.jpg");
    detector->detect(img, boxes, 0.25f);
    
    // ... 处理结果 ...
    
    delete detector;
    return 0;
}
```

## 常用参数配置

### 检测参数

```cpp
detector->detect(
    img,                // 输入图像（BGR 格式）
    boxes,              // 输出旋转框
    0.25f,              // score_threshold: 置信度阈值
    0.45f,              // iou_threshold: NMS IoU 阈值（模型已处理，此参数被忽略）
    300                 // topk: 最大检测数量
);
```

### 推荐配置

| 场景 | score_threshold | topk | 说明 |
|------|----------------|------|------|
| 高精度 | 0.5 - 0.7 | 100 | 减少误检 |
| 平衡 | 0.25 - 0.4 | 300 | 默认配置 |
| 高召回 | 0.1 - 0.2 | 500 | 检测更多目标 |

## 可视化结果

### 绘制旋转框

```cpp
// 方法 1: 直接在原图上绘制
lite::utils::draw_boxes_with_angle_inplace(img, boxes);
cv::imwrite("result.jpg", img);

// 方法 2: 保留原图，返回新图像
cv::Mat result = lite::utils::draw_boxes_with_angle(img, boxes);
cv::imwrite("result.jpg", result);
```

### 自定义绘制

```cpp
#include <opencv2/opencv.hpp>

for (const auto &box : boxes)
{
    // 计算旋转矩形的四个顶点
    cv::RotatedRect rbox(
        cv::Point2f(box.cx, box.cy),
        cv::Size2f(box.width, box.height),
        box.angle * 180.0f / CV_PI  // 转换为度数
    );
    
    cv::Point2f vertices[4];
    rbox.points(vertices);
    
    // 绘制旋转矩形
    for (int i = 0; i < 4; i++)
    {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], 
                 cv::Scalar(0, 255, 0), 2);
    }
    
    // 绘制标签
    std::string label = std::string(box.label_text) + ": " + 
                       std::to_string(box.score).substr(0, 4);
    cv::putText(img, label, vertices[0], 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                cv::Scalar(0, 0, 255), 2);
}
```

## 性能对比

| 后端 | 平台 | 推理时间 (640x640) | 相对性能 |
|------|------|-------------------|---------|
| ONNXRuntime | CPU | ~100ms | 1x |
| ONNXRuntime | GPU | ~15ms | 6.7x ⚡ |
| TensorRT FP32 | GPU | ~10ms | 10x ⚡⚡ |
| TensorRT FP16 | GPU | ~5ms | 20x ⚡⚡⚡ |

*测试环境: AMD Ryzen 9 8945HX, NVIDIA RTX 4060*

## 常见问题排查

### 问题 1: 模型加载失败

```
Error: Cannot load model file
```

**解决方案**:
- 检查模型文件路径是否正确
- 确认文件存在且有读取权限
- ONNX 模型检查：`python3 -c "import onnx; onnx.checker.check_model('model.onnx')"`

### 问题 2: 检测结果为空

```
Detected Boxes: 0
```

**解决方案**:
- 降低 `score_threshold`（如 0.1）
- 检查输入图像是否正确加载
- 确认模型与数据集匹配

### 问题 3: TensorRT Engine 加载失败

```
Error: TensorRT engine not properly initialized
```

**解决方案**:
- 在目标 GPU 上重新生成 Engine 文件
- 检查 CUDA、cuDNN、TensorRT 版本兼容性
- 确认 GPU 有足够显存

### 问题 4: 角度显示异常

**解决方案**:
```cpp
// 角度是弧度值，需要转换为度数
float degree = box.angle * 180.0 / CV_PI;

// 归一化到 [0, 360) 范围
if (degree < 0) degree += 360.0;
```

## 进阶用法

### 批量处理图像

```cpp
std::vector<std::string> image_paths = {
    "img1.jpg", "img2.jpg", "img3.jpg"
};

for (const auto &path : image_paths)
{
    cv::Mat img = cv::imread(path);
    std::vector<lite::types::BoxfWithAngle> boxes;
    
    detector->detect(img, boxes, 0.25f);
    
    // 保存结果
    std::string save_path = "result_" + 
        path.substr(path.find_last_of("/\\") + 1);
    lite::utils::draw_boxes_with_angle_inplace(img, boxes);
    cv::imwrite(save_path, img);
}
```

### 过滤特定类别

```cpp
std::vector<lite::types::BoxfWithAngle> boxes;
detector->detect(img, boxes, 0.25f);

// 只保留"BarCode"类别
std::vector<lite::types::BoxfWithAngle> barcodes;
for (const auto &box : boxes)
{
    if (std::string(box.label_text) == "BarCode")
    {
        barcodes.push_back(box);
    }
}

std::cout << "Found " << barcodes.size() << " barcodes\n";
```

### 计算旋转矩形的面积

```cpp
for (const auto &box : boxes)
{
    float area = box.width * box.height;
    std::cout << "Box area: " << area << " pixels²\n";
}
```

## 下一步

- 📖 阅读完整文档：[yolo26_obb_support.md](yolo26_obb_support.md)
- 🔧 查看示例代码：`examples/lite/cv/test_lite_yolo26_obb.cpp`
- 🚀 了解 TensorRT 优化：[TensorRT 集成指南](tensorrt/tensorrt-linux-x86_64.zh.md)
- 💡 参与贡献：[贡献指南](contrib/CONTRIBUTING.zh.md)

## 支持与反馈

如有问题或建议，欢迎通过以下方式联系：

- GitHub Issues: [lite.ai.toolkit/issues](https://github.com/DefTruth/lite.ai.toolkit/issues)
- 邮件: support@lite.ai.toolkit.org（示例）

---

**最后更新**: 2026-02-11  
**版本**: v1.0.0
