# YOLO26-OBB 支持

## 概述

lite.ai.toolkit 现已支持 YOLO26-OBB 旋转目标检测模型。YOLO26-OBB 是基于 Ultralytics YOLO 架构的 Oriented Bounding Box (OBB) 检测器，专门用于检测任意方向的目标，广泛应用于遥感图像、文档分析、场景文字检测等领域。

## 支持的后端

- **ONNXRuntime**: 完全支持，推荐使用 ✅
- **TensorRT**: 完全支持，GPU 加速推理，性能最佳 ✅
- **MNN**: 计划支持 🔜
- **NCNN**: 计划支持 🔜
- **TNN**: 计划支持 🔜

## 模型准备

### 1. 导出 ONNX 模型

使用 Ultralytics YOLO 导出 OBB ONNX 模型：

```python
from ultralytics import YOLO

# 加载 OBB 模型（确保是 -obb 版本）
model = YOLO('yolo26m-obb.pt')  # 或 yolo26n-obb, yolo26s-obb, yolo26l-obb, yolo26x-obb

# 导出为 ONNX 格式（包含 NMS 后处理，推荐）
model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)

# 查看模型信息
print(f'Task: {model.task}')  # 应该显示 'obb'
print(f'Classes: {model.names}')
```

**重要说明**：
- 导出的 ONNX 模型默认包含 NMS 后处理
- 输出格式：`[batch, num_detections, 7]`，其中 7 个值为 `[x1, y1, x2, y2, score, class_id, angle]`
- `angle` 为旋转角度（弧度值）

### 2. 转换为 TensorRT Engine

```bash
# 方法1: 使用 trtexec 转换
trtexec --onnx=yolo26m-obb.onnx \
    --saveEngine=yolo26m-obb.engine \
    --fp16 \
    --workspace=4096

# 方法2: 使用 Ultralytics 直接导出（推荐）
```

```python
from ultralytics import YOLO

model = YOLO('yolo26m-obb.pt')

# 导出为 TensorRT Engine
model.export(format='engine', device=0, half=True)  # half=True 启用 FP16
```

## 使用方法

### C++ API - ONNXRuntime

```cpp
#include "lite/lite.h"

// 使用 ONNXRuntime 后端
auto *detector = new lite::onnxruntime::cv::detection::YOLO26OBB("yolo26m-obb.onnx");

// 可选：设置自定义类别名称
std::vector<std::string> class_names = {"ExpressBillSeg", "BarCode", "2DCode"};
detector->set_class_names(class_names);

// 检测
std::vector<lite::types::BoxfWithAngle> detected_boxes;
cv::Mat img = cv::imread("test.jpg");

detector->detect(img, detected_boxes, 
                 0.25f,  // score_threshold
                 0.45f,  // iou_threshold (模型已包含 NMS，此参数被忽略)
                 300);   // topk

// 绘制旋转框
lite::utils::draw_boxes_with_angle_inplace(img, detected_boxes);
cv::imwrite("result.jpg", img);

// 访问检测结果
for (const auto &box : detected_boxes)
{
    std::cout << "Class: " << box.label_text 
              << " | Score: " << box.score
              << " | Angle: " << box.angle * 180.0 / CV_PI << "°"
              << " | Center: (" << box.cx << ", " << box.cy << ")"
              << " | Size: " << box.width << "x" << box.height << std::endl;
}

delete detector;
```

### C++ API - TensorRT

```cpp
#include "lite/lite.h"

// 使用 TensorRT 后端（GPU 加速）
auto *detector = new lite::trt::cv::detection::YOLO26OBB("yolo26m-obb.engine");

// 设置输入图像格式（可选，默认为 RGB）
detector->setInputFormat(lite::trt::cv::detection::YOLO26OBB::ImageFormat::BGR);

// 设置自定义类别名称
std::vector<std::string> class_names = {"Class0", "Class1", "Class2"};
detector->set_class_names(class_names);

// 检测
std::vector<lite::types::BoxfWithAngle> detected_boxes;
cv::Mat img = cv::imread("test.jpg");

detector->detect(img, detected_boxes, 0.25f);

// 绘制结果
lite::utils::draw_boxes_with_angle_inplace(img, detected_boxes);

delete detector;
```

### 参数说明

```cpp
void detect(
    const cv::Mat &mat,                          // 输入图像 (BGR 格式)
    std::vector<types::BoxfWithAngle> &detected_boxes,  // 输出旋转框
    float score_threshold = 0.25f,               // 置信度阈值
    float iou_threshold = 0.45f,                 // NMS IoU 阈值（已在模型中处理）
    unsigned int topk = 300                      // 最大检测数量
);
```

### 旋转框结构体

```cpp
struct BoxfWithAngle {
    float x1, y1, x2, y2;     // 轴对齐边界框（近似）
    float cx, cy;             // 旋转框中心点
    float width, height;      // 旋转框宽度和高度
    float angle;              // 旋转角度（弧度，范围通常为 [-π/4, 3π/4)）
    float score;              // 置信度分数
    unsigned int label;       // 类别 ID
    const char *label_text;   // 类别名称
    bool flag;                // 标志位
};
```

## 模型输入输出规格

### 输入
- **格式**: RGB float32
- **形状**: `[1, 3, 640, 640]`
- **预处理**: 
  - Resize with letterbox (保持宽高比)
  - 归一化到 [0, 1]
  - 通道顺序: RGB

### 输出（包含 NMS 后处理）
- **格式**: float32
- **形状**: `[1, 300, 7]`
- **值含义**: `[x1, y1, x2, y2, score, class_id, angle]`
  - `x1, y1, x2, y2`: 轴对齐边界框坐标（相对于输入图像）
  - `score`: 置信度分数 [0, 1]
  - `class_id`: 类别索引
  - `angle`: 旋转角度（弧度）

## 应用场景

YOLO26-OBB 特别适用于以下场景：

1. **遥感图像分析**
   - 飞机、船舶、车辆检测
   - 建筑物检测

2. **文档处理**
   - 表格检测
   - 文本行检测
   - 二维码/条形码检测

3. **场景文字检测**
   - 任意方向的文字区域检测
   - 街景标牌检测

4. **工业检测**
   - 零件方向识别
   - 产品缺陷检测

## 性能优化建议

### 1. 使用 TensorRT 后端
```cpp
// TensorRT 提供最佳 GPU 性能
auto *detector = new lite::trt::cv::detection::YOLO26OBB("model.engine");
```

### 2. 启用 FP16 精度
```bash
# 导出时启用半精度
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

### 3. 批量处理（未来支持）
```cpp
// 计划支持批量检测以提高吞吐量
// detector->detect_batch(images, all_boxes);
```

## 示例代码

完整示例代码可在以下位置找到：

```bash
# ONNXRuntime 示例
examples/lite/cv/test_lite_yolo26_obb.cpp

# TensorRT 示例  
examples/lite/cv/test_lite_yolo26_obb_trt.cpp
```

## 编译和测试

### 编译

```bash
cd lite.ai.toolkit
mkdir build && cd build

# 启用 ONNXRuntime 和 TensorRT
cmake -DENABLE_ONNXRUNTIME=ON \
      -DENABLE_TENSORRT=ON \
      -DENABLE_TEST=ON \
      ..

make -j$(nproc)
```

### 测试

```bash
# 准备测试数据
cp /path/to/yolo26m-obb.onnx test/model/
cp /path/to/test_image.jpg test/Data/

# 运行测试
cd build
./bin/test_lite_yolo26_obb_ort      # ONNXRuntime
./bin/test_lite_yolo26_obb_trt      # TensorRT
```

## 常见问题

### Q1: 如何理解旋转角度？
**A**: 角度值为弧度制，范围通常为 `[-π/4, 3π/4)`。可以用 `angle * 180 / π` 转换为度数。正值表示逆时针旋转。

### Q2: 如何可视化旋转框？
**A**: 使用 `lite::utils::draw_boxes_with_angle_inplace()` 函数，它会自动绘制旋转矩形。

```cpp
lite::utils::draw_boxes_with_angle_inplace(img, detected_boxes);
```

### Q3: 支持自定义类别吗？
**A**: 支持。使用 `set_class_names()` 方法设置自定义类别名称：

```cpp
detector->set_class_names({"Class1", "Class2", "Class3"});
```

### Q4: 模型输出的坐标是相对于什么的？
**A**: 坐标是相对于原始输入图像（640x640 after letterbox），框架会自动将其映射回原始图像尺寸。

### Q5: TensorRT 和 ONNXRuntime 结果一致吗？
**A**: 应该非常接近。由于浮点精度和优化策略的差异，可能存在微小差异，但不影响实际使用。

## 与 YOLOv8-OBB 的区别

| 特性 | YOLO26-OBB | YOLOv8-OBB |
|------|------------|------------|
| 架构 | 最新 YOLO26 架构 | YOLOv8 架构 |
| 输出格式 | NMS 后处理（简化） | 原始输出（需手动 NMS） |
| API 接口 | 统一简化接口 | 底层接口 |
| 性能 | 更快更准确 | 基准性能 |

## 注意事项

1. ✅ **模型格式**: 确保使用带有 `-obb` 后缀的模型
2. ✅ **输出理解**: 模型输出已包含 NMS，无需手动后处理
3. ✅ **角度范围**: 注意角度值为弧度制
4. ✅ **类别设置**: 使用自定义数据集时，记得设置正确的类别名称
5. ✅ **GPU 内存**: TensorRT 推理需要足够的 GPU 显存
6. ⚠️ **平台兼容**: TensorRT Engine 文件需要在目标 GPU 上重新生成

## 相关文档

- [YOLO12 支持文档](yolo12_support.md)
- [TensorRT 集成指南](tensorrt/tensorrt-linux-x86_64.zh.md)
- [API 参考文档](api/api.onnxruntime.md)

## 更新日志

### v1.0.0 (2026-02-11)
- ✅ 初始版本发布
- ✅ ONNXRuntime 后端支持
- ✅ TensorRT 后端支持
- ✅ 自定义类别名称支持

### 计划中的功能
- [ ] MNN 后端支持
- [ ] NCNN 后端支持
- [ ] 批量推理支持
- [ ] Python API 绑定

## 反馈与贡献

如有问题或建议，欢迎通过以下方式反馈：

- GitHub Issues: [lite.ai.toolkit/issues](https://github.com/DefTruth/lite.ai.toolkit/issues)
- Pull Requests 欢迎贡献代码

---

**许可证**: Apache 2.0  
**作者**: lite.ai.toolkit 团队  
**最后更新**: 2026-02-11
