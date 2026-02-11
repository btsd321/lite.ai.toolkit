# Lite.AI.Toolkit 项目知识总结

## 目录
- [Lite.AI.Toolkit 项目知识总结](#liteaitoolkit-项目知识总结)
  - [目录](#目录)
  - [一、项目简介](#一项目简介)
  - [二、核心特性](#二核心特性)
  - [三、技术架构](#三技术架构)
  - [四、模型生态](#四模型生态)
  - [五、扩展功能](#五扩展功能)

---

## 一、项目简介

Lite.AI.Toolkit是一个轻量级的C++ AI推理工具包，集成了100多个常用的深度学习模型，涵盖目标检测、人脸识别、图像分割、抠图等多个计算机视觉任务。项目主要面向Linux系统（Ubuntu 20.04+），提供统一简洁的API接口，最小化依赖（默认仅需OpenCV和ONNXRuntime），让开发者能够快速部署AI能力到生产环境。项目包含500+预训练权重文件，支持从基础模型到最新YOLO系列的广泛应用场景。

## 二、核心特性

**1. 简洁易用**：采用`lite::cv::Type::Class`统一语法风格，如`lite::cv::detection::YoloV5`，API设计一致性强。

**2. 最小依赖**：默认配置仅依赖OpenCV和ONNXRuntime，可选启用MNN、NCNN、TNN等其他推理引擎。

**3. 多引擎支持**：支持5种主流推理后端（ONNXRuntime、MNN、NCNN、TNN、TensorRT），其中TensorRT可充分发挥NVIDIA GPU性能，需要TensorRT 10.x和CUDA 12.x版本。

**4. 丰富模型库**：内置300+模型实现和500+预训练权重，覆盖检测、识别、分割、风格迁移等任务。

## 三、技术架构

项目采用模块化分层设计，核心目录结构包括：

- **lite/ort**：ONNXRuntime推理实现
- **lite/trt**：TensorRT GPU加速实现  
- **lite/mnn**、**lite/ncnn**、**lite/tnn**：其他引擎实现
- **lite/types.h**：统一数据类型定义（Boxf、BoxfWithLandmarks、OBBoxf等）
- **lite/models.h**：模型类型别名汇总

通过CMake配置选项（ENABLE_ONNXRUNTIME、ENABLE_TENSORRT等）灵活控制编译内容，安装包含完整的头文件和动态库，支持通过`find_package(lite.ai.toolkit)`快速集成。

## 四、模型生态

项目支持YOLO系列全版本（YOLOv3/v4/v5/v6/v8/X/R/P），特别针对人脸任务优化了YOLOv5Face、YOLOv8Face等变体。分类模型包括ResNet、DenseNet、MobileNet等主流架构；分割领域有DeepLabV3、FCN、人像分割等；抠图提供RobustVideoMatting、MODNet等实时方案；还包含人脸关键点检测（PFLD、PIPNet）、年龄性别识别、表情分析等丰富功能。模型权重可从百度网盘、Google Drive或Docker Hub获取。

## 五、扩展功能

在官方基础上新增了多项扩展：

**1. YOLOv8-OBB**：支持旋转目标检测，输出包含角度信息的定向边界框（OBBoxf），模型输出格式为`[cx, cy, w, h, cls_scores..., angle]`，类别分数已经过sigmoid激活。

**2. YOLOv11与YOLO12**：新增了最新版本YOLO系列支持，对应的TensorRT实现位于`lite/trt/cv/trt_yolov11.h`和`trt_yolo12.h`。

**3. 多任务融合流程**：提供FaceFusion Pipeline等组合应用，利用TensorRT加速实现人脸融合等复杂任务。

项目当前由社区维护，原作者专注于LLM/VLM推理优化。

## 六、代码规范
- **C++**：遵循 Google C++ Style Guide，使用 clang-format
- **命名**：snake_case（变量/函数），PascalCase（类）
- **注释**：关键函数需要 Doxygen 格式注释

## 七、主要模块
- **Detection**：包含YOLO系列、SSD、RetinaNet等目标检测模型
- **Classification**：ResNet、DenseNet、MobileNet等图像分类模型
- **Segmentation**：DeepLabV3、FCN、人像分割等语义分割模型
- **Matting**：RobustVideoMatting、MODNet等抠图模型
- **Face**：人脸检测、关键点、年龄性别识别等人脸相关模型