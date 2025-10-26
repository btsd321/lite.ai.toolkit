//
// Created by DefTruth on 2024/10/26.
//

#include "lite/trt/core/trt_utils.h"
#include "lite/utils.h"
#include "trt_yolo12.h"

using trtcv::TRTYOLO12;

void TRTYOLO12::auto_detect_nms_plugin()
{
    // 检测模型是否包含NMS插件
    // Ultralytics YOLO12 with end2end=True exports with shape [1, max_det, 6]
    // where 6 = [x1, y1, x2, y2, score, class]
    if (output_node_dims.size() >= 1)
    {
        auto &dims = output_node_dims[0];
        // Check for post-NMS output format: [batch, max_det, 6]
        if (dims.size() == 3 && dims[2] == 6)
        {
            has_nms_plugin = true;
        }
    }

#if LITETRT_DEBUG
    std::cout << "YOLO12 NMS plugin detected: "
              << (has_nms_plugin ? "Yes" : "No") << std::endl;
    if (has_nms_plugin)
    {
        std::cout << "Model has post-NMS output with "
                  << output_node_dims.size() << " outputs" << std::endl;
        for (size_t i = 0; i < output_node_dims.size(); ++i)
        {
            std::cout << "Output " << i << " shape: [";
            for (size_t j = 0; j < output_node_dims[i].size(); ++j)
            {
                std::cout << output_node_dims[i][j];
                if (j < output_node_dims[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
#endif
}

void TRTYOLO12::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size,
                          YOLO12ScaleParams &scale_params)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float height = image.rows;
    float width = image.cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh)
    {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else
    {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                       {114, 114, 114});

    // Convert to float and normalize to [0, 1]
    tmp.convertTo(out, CV_32FC3, 1.0 / 255.0);

    // Record scale params
    scale_params.r = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.new_unpad_w = padw;
    scale_params.new_unpad_h = padh;
    scale_params.flag = true;
}

void TRTYOLO12::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                               int target_height, int target_width,
                               YOLO12ScaleParams &scale_params)
{
    if (mat.empty()) return;
    cv::Size target_size(target_width, target_height);
    this->letterbox(mat, mat_rs, target_size, scale_params);
}

void TRTYOLO12::detect(const cv::Mat &mat,
                       std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, float iou_threshold,
                       unsigned int topk, unsigned int nms_type)
{
    if (mat.empty()) return;

    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // 1. Preprocess: resize and normalize
    cv::Mat processed_mat;
    YOLO12ScaleParams scale_params;
    int target_height = input_node_dims[2];
    int target_width = input_node_dims[3];
    cv::Size target_size(target_width, target_height);
    this->letterbox(mat, processed_mat, target_size, scale_params);

#if LITETRT_DEBUG
    std::cout << "Input image: " << mat.rows << "x" << mat.cols << std::endl;
    std::cout << "Processed image: " << processed_mat.rows << "x"
              << processed_mat.cols << " channels=" << processed_mat.channels()
              << " type=" << processed_mat.type() << std::endl;
    std::cout << "Target size: " << target_width << "x" << target_height
              << std::endl;
    std::cout << "Expected input dims: [" << input_node_dims[0] << ", "
              << input_node_dims[1] << ", " << input_node_dims[2] << ", "
              << input_node_dims[3] << "]" << std::endl;
#endif

    // 2. Make input tensor
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(
        processed_mat, input, input_node_dims, trtcv::utils::transform::CHW);

#if LITETRT_DEBUG
    size_t expected_size = input_node_dims[0] * input_node_dims[1] *
                           input_node_dims[2] * input_node_dims[3];
    std::cout << "Created tensor size: " << input.size()
              << ", expected: " << expected_size << std::endl;
    if (!input.empty())
    {
        std::cout << "First few values: " << input[0] << ", " << input[1]
                  << ", " << input[2] << std::endl;
    }
#endif

    // 3. Copy to GPU memory
    size_t input_size = input_node_dims[0] * input_node_dims[1] *
                        input_node_dims[2] * input_node_dims[3] * sizeof(float);
    cudaError_t cuda_status = cudaMemcpyAsync(
        buffers[0], input.data(), input_size, cudaMemcpyHostToDevice, stream);
    if (cuda_status != cudaSuccess)
    {
#if LITETRT_DEBUG
        std::cout << "CUDA memcpy failed: " << cudaGetErrorString(cuda_status)
                  << std::endl;
#endif
        return;
    }
    cudaStreamSynchronize(stream);

#if LITETRT_DEBUG
    std::cout << "Data copied to GPU successfully" << std::endl;
#endif

    // 4. Inference
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    if (!status)
    {
#if LITETRT_DEBUG
        std::cout << "TensorRT inference failed!" << std::endl;
#endif
        return;
    }

    cudaStreamSynchronize(stream);

    // 5. Get output dimensions
    auto pred_dims = output_node_dims[0];

    // 6. Copy outputs from GPU and parse
    if (has_nms_plugin)
    {
        // Ultralytics end2end=True model - output already has NMS applied
        this->generate_bboxes_with_nms(scale_params, detected_boxes,
                                       score_threshold, img_height, img_width);
    }
    else
    {
        // Standard model - need to apply NMS
        std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);
        cudaMemcpyAsync(
            output.data(), buffers[1],
            pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 生成检测框
        std::vector<types::Boxf> bbox_collection;
        this->generate_bboxes(scale_params, bbox_collection, output.data(),
                              score_threshold, img_height, img_width);

        // NMS
        this->nms(bbox_collection, detected_boxes, iou_threshold, topk,
                  nms_type);
    }
}

void TRTYOLO12::generate_bboxes_with_nms(
    const YOLO12ScaleParams &scale_params,
    std::vector<types::Boxf> &bbox_collection, float score_threshold,
    int img_height, int img_width)
{
    // 处理Ultralytics end2end=True模型的输出
    // 输出格式：[batch, max_det, 6] where 6 = [x1, y1, x2, y2, score, class]
    bbox_collection.clear();

    if (output_node_dims.size() < 1)
    {
#if LITETRT_DEBUG
        std::cout << "Warning: No output node dims available" << std::endl;
#endif
        return;
    }

    auto &dims = output_node_dims[0];
    if (dims.size() != 3 || dims[2] != 6)
    {
#if LITETRT_DEBUG
        std::cout << "Warning: Unexpected output format for NMS model"
                  << std::endl;
#endif
        return;
    }

    // 从GPU复制输出
    const int max_det = dims[1];
    std::vector<float> output(max_det * 6);
    cudaMemcpyAsync(output.data(), buffers[1], max_det * 6 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float r_ = scale_params.r;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    // 解析检测结果
    for (int i = 0; i < max_det; ++i)
    {
        float *det = &output[i * 6];
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float score = det[4];
        float class_id = det[5];

        // 跳过无效检测（score为0或负数）
        if (score <= score_threshold) continue;

        // 反归一化坐标
        x1 = (x1 - dw_) / r_;
        y1 = (y1 - dh_) / r_;
        x2 = (x2 - dw_) / r_;
        y2 = (y2 - dh_) / r_;

        // 裁剪到图像边界
        x1 = std::max(std::min(x1, (float)img_width - 1.0f), 0.0f);
        y1 = std::max(std::min(y1, (float)img_height - 1.0f), 0.0f);
        x2 = std::max(std::min(x2, (float)img_width - 1.0f), 0.0f);
        y2 = std::max(std::min(y2, (float)img_height - 1.0f), 0.0f);

        types::Boxf box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.score = score;
        box.label = static_cast<unsigned int>(class_id);
        box.label_text = (box.label < 80) ? class_names[box.label] : "unknown";
        box.flag = true;
        bbox_collection.push_back(box);
    }

#if LITETRT_DEBUG
    std::cout << "Detected " << bbox_collection.size()
              << " objects from NMS output" << std::endl;
#endif
}

void TRTYOLO12::generate_bboxes(const YOLO12ScaleParams &scale_params,
                                std::vector<types::Boxf> &bbox_collection,
                                float *output, float score_threshold,
                                int img_height, int img_width)
{
    // Output format: [1, num_anchors, 84] where 84 = 4(bbox) + 80(classes)
    const unsigned int num_anchors = output_node_dims[0][1];
    const unsigned int num_classes = output_node_dims[0][2] - 4;  // 80

    float r_ = scale_params.r;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    bbox_collection.clear();
    unsigned int count = 0;

    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        float *row_ptr = output + (i * (num_classes + 4));

        // Get bbox coordinates
        float cx = row_ptr[0];
        float cy = row_ptr[1];
        float w = row_ptr[2];
        float h = row_ptr[3];

        // Find max class score
        float max_class_score = 0.0f;
        unsigned int max_class_id = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float class_score = row_ptr[j + 4];
            if (class_score > max_class_score)
            {
                max_class_score = class_score;
                max_class_id = j;
            }
        }

        if (max_class_score < score_threshold) continue;

        // Convert center format to corner format and apply scale params
        float x1 = ((cx - w / 2.0f) - dw_) / r_;
        float y1 = ((cy - h / 2.0f) - dh_) / r_;
        float x2 = ((cx + w / 2.0f) - dw_) / r_;
        float y2 = ((cy + h / 2.0f) - dh_) / r_;

        // Clamp to image boundaries
        x1 = std::max(std::min(x1, (float)img_width - 1.0f), 0.0f);
        y1 = std::max(std::min(y1, (float)img_height - 1.0f), 0.0f);
        x2 = std::max(std::min(x2, (float)img_width - 1.0f), 0.0f);
        y2 = std::max(std::min(y2, (float)img_height - 1.0f), 0.0f);

        types::Boxf box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.score = max_class_score;
        box.label = max_class_id;
        box.label_text = class_names[max_class_id];
        box.flag = true;
        bbox_collection.push_back(box);

        count += 1;
        if (count > max_nms) break;
    }

#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << std::endl;
    std::cout << "generate_bboxes num: " << bbox_collection.size() << std::endl;
#endif
}

void TRTYOLO12::nms(std::vector<types::Boxf> &input,
                    std::vector<types::Boxf> &output, float iou_threshold,
                    unsigned int topk, unsigned int nms_type)
{
    if (nms_type == NMS::BLEND)
    {
        lite::utils::blending_nms(input, output, iou_threshold, topk);
    }
    else if (nms_type == NMS::OFFSET)
    {
        lite::utils::offset_nms(input, output, iou_threshold, topk);
    }
    else
    {
        lite::utils::hard_nms(input, output, iou_threshold, topk);
    }
}