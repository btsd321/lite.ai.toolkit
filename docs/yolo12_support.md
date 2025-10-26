# YOLO12 支持

## 概述

lite.ai.toolkit 现已支持 YOLO12 目标检测模型。YOLO12 是 Ultralytics 推出的最新版本目标检测模型，具有更高的精度和更快的推理速度。

## 支持的后端

- **ONNXRuntime**: 完全支持，推荐使用
- **TensorRT**: 完全支持，GPU 加速推理，性能最佳
- **MNN**: 支持
- **NCNN**: 支持  
- **TNN**: 基础支持

## 模型准备

### 1. 导出 ONNX 模型

使用 Ultralytics YOLO12 导出 ONNX 模型：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo12n.pt')  # yolo12n, yolo12s, yolo12m, yolo12l, yolo12x

# 导出为 ONNX 格式
model.export(format='onnx', opset=11, simplify=True, dynamic=False, imgsz=640)
```

### 2. 转换其他格式

根据需要将 ONNX 模型转换为其他推理后端格式：

```bash
# 转换为 TensorRT Engine (推荐用于 GPU 部署)
trtexec --onnx=yolo12n.onnx --saveEngine=yolo12n.engine --fp16

# 转换为 MNN
./MNNConvert -f ONNX --modelFile yolo12n.onnx --MNNModel yolo12n.mnn --bizCode MNN

# 转换为 NCNN (需要 onnx2ncnn 工具)
onnx2ncnn yolo12n.onnx yolo12n.param yolo12n.bin

# 转换为 TNN (需要 onnx2tnn 工具)
onnx2tnn yolo12n.onnx -o ./
```

## 使用方法

### C++ API

```cpp
#include "lite/lite.h"

// 使用默认后端 (ONNXRuntime)
lite::cv::detection::YOLO12 *yolo12 = new lite::cv::detection::YOLO12("yolo12n.onnx");

// 或指定具体后端
// ONNXRuntime
lite::onnxruntime::cv::detection::YOLO12 *yolo12_ort = 
    new lite::onnxruntime::cv::detection::YOLO12("yolo12n.onnx");

// TensorRT (GPU 加速，推荐)
lite::trt::cv::detection::YOLO12 *yolo12_trt = 
    new lite::trt::cv::detection::YOLO12("yolo12n.engine");

// MNN
lite::mnn::cv::detection::YOLO12 *yolo12_mnn = 
    new lite::mnn::cv::detection::YOLO12("yolo12n.mnn");

// NCNN
lite::ncnn::cv::detection::YOLO12 *yolo12_ncnn = 
    new lite::ncnn::cv::detection::YOLO12("yolo12n.param", "yolo12n.bin");

// 检测
std::vector<lite::types::Boxf> detected_boxes;
cv::Mat img = cv::imread("test.jpg");
yolo12->detect(img, detected_boxes, 0.25f, 0.45f, 100, lite::cv::detection::YOLO12::NMS::OFFSET);

// 绘制结果
lite::utils::draw_boxes_inplace(img, detected_boxes);
cv::imwrite("result.jpg", img);

delete yolo12;
```

### 参数说明

- `score_threshold`: 置信度阈值，默认 0.25
- `iou_threshold`: NMS IoU 阈值，默认 0.45  
- `topk`: 最大检测框数量，默认 100
- `nms_type`: NMS 类型，支持 HARD/BLEND/OFFSET，默认 OFFSET

## 性能特点

YOLO12 相比之前版本具有以下特点：

1. **更高精度**: 采用了新的网络架构设计
2. **更快速度**: 优化的推理管道和网络结构
3. **更好的小目标检测**: 改进的特征融合策略
4. **更强的泛化能力**: 增强的数据增强和训练策略

## 模型规格

| 模型 | 参数量 | GFLOPs | mAP50-95 | 输入尺寸 |
|------|--------|--------|----------|----------|
| YOLO12n | 2.6M | 6.7 | - | 640x640 |
| YOLO12s | 9.3M | 21.7 | - | 640x640 |
| YOLO12m | 20.2M | 68.1 | - | 640x640 |
| YOLO12l | 26.5M | 89.7 | - | 640x640 |
| YOLO12x | 59.2M | 200.3 | - | 640x640 |

## 编译和测试

### 编译

```bash
cd lite.ai.toolkit
mkdir build && cd build
cmake -DENABLE_ONNXRUNTIME=ON -DENABLE_TENSORRT=ON -DENABLE_TEST=ON ..
make -j4
```

### 测试

```bash
cd build
make install

# 测试所有后端
./bin/test_lite_yolo12

# 或测试特定后端
./bin/test_lite_yolo12_ort     # ONNXRuntime
./bin/test_lite_yolo12_trt     # TensorRT
./bin/test_lite_yolo12_mnn     # MNN
./bin/test_lite_yolo12_ncnn    # NCNN
```

## 注意事项

1. 确保模型输入尺寸为 640x640 (默认)
2. 支持动态输入尺寸，但需要在导出时设置
3. 推荐使用 TensorRT 后端以获得最佳 GPU 性能
4. ONNXRuntime 后端提供最佳兼容性
5. 模型文件路径需要正确设置
6. TensorRT Engine 文件与 GPU 架构相关，需要在目标设备上生成

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型文件路径和格式
2. **推理结果异常**: 确认模型输入预处理正确
3. **性能不佳**: 尝试不同的后端或优化设置

### 调试选项

编译时启用调试信息：

```bash
cmake -DENABLE_DEBUG_STRING=ON -DENABLE_ONNXRUNTIME=ON ..
```

这将输出详细的调试信息，帮助定位问题。

## 下一步计划

- [x] 完成 TNN 后端的完整实现
- [x] 添加 TensorRT 后端支持
- [ ] 优化推理性能
- [ ] 添加批量推理支持
- [ ] 支持更多的 YOLO12 变体模型