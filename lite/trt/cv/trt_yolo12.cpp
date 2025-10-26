//
// Created by DefTruth on 2024/10/26.
//

#include "trt_yolo12.h"
#include "lite/trt/core/trt_utils.h"
#include "lite/utils.h"

using trtcv::TRTYOLO12;

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

void TRTYOLO12::preprocess(const cv::Mat &input_image, cv::Mat &output_mat)
{
    YOLO12ScaleParams scale_params;
    this->resize_unscale(input_image, output_mat, input_height, input_width, scale_params);
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
    cv::Size target_size(input_width, input_height);
    this->letterbox(mat, processed_mat, target_size, scale_params);

    // 2. Copy to GPU memory
    auto input_size = processed_mat.total() * processed_mat.elemSize();
    cudaMemcpyAsync(device_ptrs[0], processed_mat.ptr<float>(), input_size,
                    cudaMemcpyHostToDevice, stream);

    // 3. Inference
    if (!context->executeV2(device_ptrs.data()))
    {
#ifdef LITETRT_DEBUG
        std::cout << "TensorRT inference failed!" << std::endl;
#endif
        return;
    }

    // 4. Copy output from GPU
    auto output_size = output_node_sizes[0] * sizeof(float);
    cudaMemcpyAsync(host_ptrs[0], device_ptrs[num_inputs], output_size,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. Post-process: generate bboxes
    std::vector<types::Boxf> bbox_collection;
    this->generate_bboxes(scale_params, bbox_collection,
                          static_cast<float *>(host_ptrs[0]),
                          score_threshold, img_height, img_width);

    // 6. NMS
    this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
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