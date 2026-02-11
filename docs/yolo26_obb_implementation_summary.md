# YOLO26-OBB 集成总结

## 📋 项目概览

本次工作成功为 lite.ai.toolkit 添加了 YOLO26-OBB（旋转目标检测）支持，包括 ONNXRuntime 和 TensorRT 两个后端实现。

**完成时间**: 2026-02-11  
**实现者**: lite.ai.toolkit 团队

---

## 🎯 实现目标

✅ 添加 YOLO26-OBB 旋转目标检测支持  
✅ 实现 ONNXRuntime 后端  
✅ 实现 TensorRT 后端  
✅ 注册到框架模型系统  
✅ 编写完整使用文档  
✅ 创建测试示例

---

## 📁 新增文件清单

### 1. 核心实现文件

#### ONNXRuntime 后端
- `lite/ort/cv/yolo26_obb.h` - ONNXRuntime 版本头文件
- `lite/ort/cv/yolo26_obb.cpp` - ONNXRuntime 版本实现

#### TensorRT 后端
- `lite/trt/cv/trt_yolo26_obb.h` - TensorRT 版本头文件
- `lite/trt/cv/trt_yolo26_obb.cpp` - TensorRT 版本实现

### 2. 文档文件

- `docs/yolo26_obb_support.md` - 完整技术文档
- `docs/yolo26_obb_quickstart.md` - 快速入门指南
- `docs/yolo26_obb_implementation_summary.md` - 本实现总结（当前文件）

### 3. 测试文件

- `examples/lite/cv/test_lite_yolo26_obb.cpp` - 综合测试示例

---

## 🔧 修改的文件

### 1. 模型注册文件

**文件**: `lite/models.h`

**修改内容**:
1. 添加头文件引用：
   ```cpp
   #include "lite/ort/cv/yolo26_obb.h"
   #include "lite/trt/cv/trt_yolo26_obb.h"
   ```

2. 添加 ONNXRuntime typedef：
   ```cpp
   typedef ortcv::YOLO26OBB _ONNXYOLO26OBB;
   ```

3. 添加 TensorRT typedef：
   ```cpp
   typedef trtcv::TRTYOLO26OBB _TRT_YOLO26OBB;
   ```

4. 在命名空间中暴露类型：
   ```cpp
   // ONNXRuntime namespace
   namespace lite::onnxruntime::cv::detection {
       typedef _ONNXYOLO26OBB YOLO26OBB;
   }
   
   // TensorRT namespace
   namespace lite::trt::cv::detection {
       typedef _TRT_YOLO26OBB YOLO26OBB;
   }
   ```

---

## 🏗️ 架构设计

### 类层次结构

```
BasicOrtHandler (基类)
    └── YOLO26OBB (ONNXRuntime 实现)

BasicTRTHandler (基类)
    └── TRTYOLO26OBB (TensorRT 实现)
```

### 核心接口

```cpp
class YOLO26OBB : public BasicOrtHandler
{
public:
    // 构造函数
    explicit YOLO26OBB(const std::string &_onnx_path, unsigned int _num_threads = 1);
    
    // 设置自定义类别名称
    void set_class_names(const std::vector<std::string> &names);
    
    // 检测接口
    void detect(const cv::Mat &mat, 
                std::vector<types::BoxfWithAngle> &detected_boxes,
                float score_threshold = 0.25f, 
                float iou_threshold = 0.45f,
                unsigned int topk = 300);
};
```

### 数据流程

```
输入图像 (BGR)
    ↓
Resize + Letterbox (保持宽高比)
    ↓
色彩空间转换 (BGR → RGB)
    ↓
归一化 [0, 255] → [0, 1]
    ↓
ONNX/TensorRT 推理
    ↓
输出解析 [batch, 300, 7]
    ↓
坐标逆变换（回到原始图像尺寸）
    ↓
返回旋转框结果
```

---

## 🔍 技术细节

### 1. 模型输入输出

**输入**:
- 形状: `[1, 3, 640, 640]`
- 类型: float32
- 格式: RGB, 归一化到 [0, 1]

**输出**:
- 形状: `[1, 300, 7]`
- 类型: float32
- 格式: `[x1, y1, x2, y2, score, class_id, angle]`
  - `x1, y1, x2, y2`: 轴对齐边界框坐标（640x640 空间）
  - `score`: 置信度 [0, 1]
  - `class_id`: 类别索引
  - `angle`: 旋转角度（弧度）

### 2. 预处理策略

采用 Letterbox 策略保持图像宽高比：

```cpp
// 计算缩放比例
float w_r = target_width / img_width;
float h_r = target_height / img_height;
float r = std::min(w_r, h_r);

// 计算缩放后尺寸
int new_w = img_width * r;
int new_h = img_height * r;

// 计算填充
int dw = (target_width - new_w) / 2;
int dh = (target_height - new_h) / 2;
```

### 3. 后处理流程

1. **解析输出**: 遍历 300 个检测结果  2. **阈值过滤**: score >= score_threshold
3. **坐标逆变换**: 从 640x640 映射回原图  4. **边界约束**: 确保坐标在图像范围内
5. **Top-K 筛选**: 按置信度排序，保留前 K 个

### 4. 旋转框表示

使用 `BoxfWithAngle` 结构体：

```cpp
struct BoxfWithAngle {
    float x1, y1, x2, y2;     // 外接矩形（方便可视化）
    float cx, cy;             // 旋转框中心
    float width, height;      // 旋转框尺寸
    float angle;              // 旋转角度（弧度）
    float score;              // 置信度
    unsigned int label;       // 类别 ID
    const char *label_text;   // 类别名称
    bool flag;                // 有效标志
};
```

---

## 🚀 性能特性

### 1. 优化点

- ✅ NMS 已在模型中完成（端到端推理）
- ✅ 支持 GPU 加速（TensorRT）
- ✅ 支持 FP16 精度（TensorRT）
- ✅ Letterbox 策略减少形变
- ✅ 向量化操作

### 2. 内存管理

- ONNXRuntime: 自动内存管理
- TensorRT: CUDA 统一内存，异步拷贝

### 3. 线程安全

- 每个实例独立
- 支持多线程推理（创建多个实例）

---

## 📚 使用示例

### 基础用法

```cpp
#include "lite/lite.h"

int main()
{
    // 创建检测器
    auto *detector = new lite::onnxruntime::cv::detection::YOLO26OBB(
        "yolo26m-obb.onnx");
    
    // 设置类别
    detector->set_class_names({"Class0", "Class1", "Class2"});
    
    // 检测
    cv::Mat img = cv::imread("test.jpg");
    std::vector<lite::types::BoxfWithAngle> boxes;
    detector->detect(img, boxes, 0.25f);
    
    // 可视化
    lite::utils::draw_boxes_with_angle_inplace(img, boxes);
    cv::imwrite("result.jpg", img);
    
    delete detector;
}
```

### TensorRT 加速

```cpp
// 导出 TensorRT Engine
from ultralytics import YOLO
model = YOLO('yolo26m-obb.pt')
model.export(format='engine', device=0, half=True)

// C++ 使用
auto *detector = new lite::trt::cv::detection::YOLO26OBB("model.engine");
detector->setInputFormat(lite::trt::cv::detection::YOLO26OBB::ImageFormat::BGR);
// ... 其余相同
```

---

## ✅ 测试验证

### 单元测试

```bash
cd build
./bin/test_lite_yolo26_obb onnx  # 测试 ONNXRuntime
./bin/test_lite_yolo26_obb trt   # 测试 TensorRT
./bin/test_lite_yolo26_obb all   # 测试所有后端
```

### 预期输出

```
=== ONNXRuntime Version Results ===
Detected Boxes: 15
  [0] BarCode | Score: 0.892 | Angle: 15.3° | Box: [120, 80, 250, 150]
  [1] 2DCode | Score: 0.856 | Angle: -5.7° | Box: [300, 200, 380, 280]
  ...
Result saved to: test_onnx_yolo26_obb.jpg
```

---

## 🔜 未来计划

### 短期计划
- [ ] 添加 MNN 后端支持
- [ ] 添加 NCNN 后端支持
- [ ] 优化预处理性能
- [ ] 添加批量推理支持

### 长期计划
- [ ] Python API 绑定
- [ ] 模型量化支持（INT8）
- [ ] 动态输入尺寸支持
- [ ] ONNX 模型简化工具

---

## 📝 开发注意事项

### 1. 添加新后端

参考 ONNXRuntime 和 TensorRT 实现：

1. 创建 `lite/{backend}/cv/{backend}_yolo26_obb.h`
2. 创建 `lite/{backend}/cv/{backend}_yolo26_obb.cpp`
3. 在 `models.h` 中注册
4. 实现 `detect()` 接口
5. 添加测试用例

### 2. 调试技巧

启用调试输出：

```cpp
// 编译时定义
#define LITEORT_DEBUG
#define LITETRT_DEBUG

// 或在 CMakeLists.txt 中
add_definitions(-DLITEORT_DEBUG -DLITETRT_DEBUG)
```

### 3. 常见陷阱

- ⚠️ 注意角度单位（弧度 vs 度数）
- ⚠️ 坐标系统（模型空间 vs 图像空间）
- ⚠️ 内存泄漏（记得 delete）
- ⚠️ GPU 显存管理（TensorRT）

---

## 📖 相关文档

1. [YOLO26-OBB 完整文档](yolo26_obb_support.md)
2. [快速入门指南](yolo26_obb_quickstart.md)
3. [YOLO12 支持文档](yolo12_support.md)（参考）
4. [TensorRT 集成指南](tensorrt/tensorrt-linux-x86_64.zh.md)

---

## 🙏 致谢

- Ultralytics YOLO 团队 - 提供优秀的 OBB 检测模型
- lite.ai.toolkit 社区 - 框架基础设施
- OpenCV 项目 - 旋转矩形可视化支持

---

## 📄 许可证

本实现遵循 lite.ai.toolkit 项目的 Apache 2.0 许可证。

---

**文档版本**: 1.0.0  
**最后更新**: 2026-02-11  
**维护者**: lite.ai.toolkit 团队
