//
// Created by DefTruth on 2024/10/26.
//

#include "trt_yolo12.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/utils.h"

using trtcv::TRTYOLO12;

void TRTYOLO12::auto_detect_nms_plugin()
{
    // 检测模型是否包含NMS插件
    // 通常自带NMS的模型会有多个输出：num_dets, det_boxes, det_scores, det_classes
    // 而标准模型只有一个输出
    if (output_node_dims.size() >= 3)
    {
        // 检查输出的形状特征
        bool likely_nms = false;
        for (const auto &dims : output_node_dims)
        {
            // NMS输出通常包含固定数量的检测框（如100）
            if (dims.size() >= 2 && (dims[1] == 100 || dims[1] == 300 || dims[1] == 1000))
            {
                likely_nms = true;
                break;
            }
        }
        has_nms_plugin = likely_nms;
    }

#ifdef LITETRT_DEBUG
    std::cout << "YOLO12 NMS plugin detected: " << (has_nms_plugin ? "Yes" : "No") << std::endl;
    if (has_nms_plugin)
    {
        std::cout << "Model has " << output_node_dims.size() << " outputs" << std::endl;
        for (size_t i = 0; i < output_node_dims.size(); ++i)
        {
            std::cout << "Output " << i << " shape: [";
            for (size_t j = 0; j < output_node_dims[i].size(); ++j)
            {
                std::cout << output_node_dims[i][j];
                if (j < output_node_dims[i].size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
#endif
}

void TRTYOLO12::letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size, YOLO12ScaleParams &scale_params)
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

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    // Convert to float and normalize
    tmp.convertTo(tmp, CV_32F, 1.0 / 255.0);

    // Convert from HWC to CHW format
    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float *)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float *)out.data + (int)inp_h * (int)inp_w * 2);

    // BGR to RGB conversion
    channels[2].copyTo(c0); // R
    channels[1].copyTo(c1); // G
    channels[0].copyTo(c2); // B

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
    if (mat.empty())
        return;
    cv::Size target_size(target_width, target_height);
    this->letterbox(mat, mat_rs, target_size, scale_params);
}

void TRTYOLO12::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, float iou_threshold,
                       unsigned int topk, unsigned int nms_type)
{
    if (mat.empty())
        return;

    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // 1. Preprocess: resize and normalize
    cv::Mat processed_mat;
    YOLO12ScaleParams scale_params;
    int target_height = input_node_dims[2];
    int target_width = input_node_dims[3];
    cv::Size target_size(target_width, target_height);
    this->letterbox(mat, processed_mat, target_size, scale_params);

    // 2. Make input tensor
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(processed_mat, input, input_node_dims, trtcv::utils::transform::CHW);

    // 3. Copy to GPU memory
    cudaMemcpyAsync(buffers[0], input.data(),
                    input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // 4. Inference
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    if (!status)
    {
#ifdef LITETRT_DEBUG
        std::cout << "TensorRT inference failed!" << std::endl;
#endif
        return;
    }

    cudaStreamSynchronize(stream);

    // 5. Get output dimensions
    auto pred_dims = output_node_dims[0];

    // 6. Copy outputs from GPU
    if (has_nms_plugin)
    {
        // NMS模型有多个输出
        // 这里需要根据实际NMS插件的输出格式处理
        // 暂时使用标准输出处理
        std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);
        cudaMemcpyAsync(output.data(), buffers[1], pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 生成检测框（NMS版本）
        this->generate_bboxes_with_nms(scale_params, detected_boxes, score_threshold, img_height, img_width);
    }
    else
    {
        // 标准模型处理
        std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);
        cudaMemcpyAsync(output.data(), buffers[1], pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // 生成检测框
        std::vector<types::Boxf> bbox_collection;
        this->generate_bboxes(scale_params, bbox_collection, output.data(),
                              score_threshold, img_height, img_width);

        // NMS
        this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
    }
}

void TRTYOLO12::generate_bboxes_with_nms(const YOLO12ScaleParams &scale_params,
                                         std::vector<types::Boxf> &bbox_collection,
                                         float score_threshold, int img_height, int img_width)
{
    // 处理包含NMS插件的模型输出
    // 典型的输出格式：
    // output[0]: num_dets [batch]
    // output[1]: det_boxes [batch, max_det, 4]
    // output[2]: det_scores [batch, max_det]
    // output[3]: det_classes [batch, max_det]

    bbox_collection.clear();

    if (output_node_dims.size() < 4)
    {
#ifdef LITETRT_DEBUG
        std::cout << "Warning: Expected 4 outputs for NMS model, got " << output_node_dims.size() << std::endl;
#endif
        // TODO: 当前版本暂不支持NMS插件，回退到标准处理
        return;
    }

    // TODO: 实现NMS插件输出的解析
    // 当前版本需要用户使用标准模型（不带nms=True）

#ifdef LITETRT_DEBUG
    std::cout << "NMS plugin model support is under development. Please use standard model without nms=True." << std::endl;
#endif
}

void TRTYOLO12::generate_bboxes(const YOLO12ScaleParams &scale_params,
                                std::vector<types::Boxf> &bbox_collection,
                                float *output,
                                float score_threshold, int img_height, int img_width)
{
    // Output format: [1, num_anchors, 84] where 84 = 4(bbox) + 80(classes)
    const unsigned int num_anchors = output_node_dims[0][1];
    const unsigned int num_classes = output_node_dims[0][2] - 4; // 80

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

        if (max_class_score < score_threshold)
            continue;

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
        if (count > max_nms)
            break;
    }

#ifdef LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << std::endl;
    std::cout << "generate_bboxes num: " << bbox_collection.size() << std::endl;
#endif
}

void TRTYOLO12::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk, unsigned int nms_type)
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