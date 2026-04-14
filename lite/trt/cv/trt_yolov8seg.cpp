//
// Created for YOLOv8-Seg TensorRT support
//

#include "trt_yolov8seg.h"
#include <opencv2/opencv.hpp>

using trtcv::TRTYoloV8Seg;

void TRTYoloV8Seg::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                   int target_height, int target_width,
                                   YOLOv8SegScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = mat.rows;
    int img_width  = mat.cols;

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    float w_r = (float)target_width  / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r   = std::min(w_r, h_r);

    int new_unpad_w = static_cast<int>((float)img_width  * r);
    int new_unpad_h = static_cast<int>((float)img_height * r);
    int dw = (target_width  - new_unpad_w) / 2;
    int dh = (target_height - new_unpad_h) / 2;

    cv::Mat new_unpad_mat;
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    scale_params.r           = r;
    scale_params.dw          = dw;
    scale_params.dh          = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag        = true;
}

void TRTYoloV8Seg::generate_detections(
    const YOLOv8SegScaleParams &scale_params,
    std::vector<types::Boxf> &bbox_collection,
    std::vector<std::vector<float>> &mask_coeffs_collection,
    const float *det_output,
    float score_threshold, int img_height, int img_width)
{
    // TRT output0: (1, total_channels, n_anchors) channel-first
    auto pred_dims = output_node_dims[0]; // (1, 116, 8400)
    const unsigned int n_anchors      = static_cast<unsigned int>(pred_dims[2]);
    const unsigned int total_channels = static_cast<unsigned int>(pred_dims[1]);
    const unsigned int num_classes    = total_channels - 4 - static_cast<unsigned int>(num_mask_coeffs);

    float r_  = scale_params.r;
    int   dw_ = scale_params.dw;
    int   dh_ = scale_params.dh;

    bbox_collection.clear();
    mask_coeffs_collection.clear();
    unsigned int count = 0;

    for (unsigned int i = 0; i < n_anchors; ++i)
    {
        // Find best class score
        float max_cls_conf = -1.f;
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float s = det_output[(4 + j) * n_anchors + i];
            if (s > max_cls_conf)
            {
                max_cls_conf = s;
                label = j;
            }
        }
        if (max_cls_conf < score_threshold) continue;

        float cx = det_output[0 * n_anchors + i];
        float cy = det_output[1 * n_anchors + i];
        float w  = det_output[2 * n_anchors + i];
        float h  = det_output[3 * n_anchors + i];

        float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float)dh_) / r_;

        x1 = std::max(std::min(x1, (float)img_width  - 1.f), 0.f);
        y1 = std::max(std::min(y1, (float)img_height - 1.f), 0.f);
        x2 = std::max(std::min(x2, (float)img_width  - 1.f), 0.f);
        y2 = std::max(std::min(y2, (float)img_height - 1.f), 0.f);

        types::Boxf box;
        box.x1         = x1;
        box.y1         = y1;
        box.x2         = x2;
        box.y2         = y2;
        box.score      = max_cls_conf;
        box.label      = label;
        box.label_text = class_names[label];
        box.flag       = true;
        bbox_collection.push_back(box);

        std::vector<float> coeffs(num_mask_coeffs);
        for (int k = 0; k < num_mask_coeffs; ++k)
            coeffs[k] = det_output[(4 + num_classes + k) * n_anchors + i];
        mask_coeffs_collection.push_back(coeffs);

        if (++count > max_nms) break;
    }

#if LITETRT_DEBUG
    std::cout << "TRTYoloV8Seg generate_detections num: " << bbox_collection.size() << "\n";
#endif
}

void TRTYoloV8Seg::nms(
    std::vector<types::Boxf> &input,
    std::vector<types::Boxf> &output,
    std::vector<std::vector<float>> &coeffs_in,
    std::vector<std::vector<float>> &coeffs_out,
    float iou_threshold, unsigned int topk)
{
    if (input.empty()) return;

    // Sort by score descending
    std::vector<unsigned int> idx(input.size());
    for (unsigned int i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](unsigned int a, unsigned int b) { return input[a].score > input[b].score; });

    std::vector<types::Boxf> sorted_boxes;
    std::vector<std::vector<float>> sorted_coeffs;
    for (auto id : idx)
    {
        sorted_boxes.push_back(input[id]);
        sorted_coeffs.push_back(coeffs_in[id]);
    }

    std::vector<bool> suppressed(sorted_boxes.size(), false);
    output.clear();
    coeffs_out.clear();

    for (unsigned int i = 0; i < sorted_boxes.size(); ++i)
    {
        if (suppressed[i]) continue;
        output.push_back(sorted_boxes[i]);
        coeffs_out.push_back(sorted_coeffs[i]);
        if (output.size() >= topk) break;

        for (unsigned int j = i + 1; j < sorted_boxes.size(); ++j)
        {
            if (suppressed[j]) continue;
            float iou = sorted_boxes[i].iou_of(sorted_boxes[j]);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
}

cv::Mat TRTYoloV8Seg::decode_single_mask(
    const std::vector<float> &coeffs,
    const float *proto_data,
    int proto_c, int proto_h, int proto_w,
    const YOLOv8SegScaleParams &scale_params,
    int img_h, int img_w,
    float x1, float y1, float x2, float y2)
{
    int proto_area = proto_h * proto_w;

    // 1. mask = coeffs @ proto.reshape(proto_c, proto_h*proto_w)
    std::vector<float> mask_vec(proto_area, 0.f);
    for (int k = 0; k < proto_c; ++k)
    {
        float ck = coeffs[k];
        const float *proto_k = proto_data + k * proto_area;
        for (int j = 0; j < proto_area; ++j)
            mask_vec[j] += ck * proto_k[j];
    }

    // 2. Sigmoid
    for (auto &v : mask_vec)
        v = 1.f / (1.f + std::exp(-v));

    // 3. Proto-space mask
    cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, mask_vec.data());

    // 4. Upsample to letterboxed input size
    int input_h = static_cast<int>(input_node_dims[2]);
    int input_w = static_cast<int>(input_node_dims[3]);
    cv::Mat mask_input;
    cv::resize(mask_proto, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    // 5. Crop letterbox padding
    cv::Mat mask_crop = mask_input(
        cv::Rect(scale_params.dw, scale_params.dh,
                 scale_params.new_unpad_w, scale_params.new_unpad_h)).clone();

    // 6. Resize to original image size
    cv::Mat mask_orig;
    cv::resize(mask_crop, mask_orig, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    // 7. Apply bounding box region + threshold
    cv::Mat result = cv::Mat::zeros(img_h, img_w, CV_32FC1);
    int bx1 = std::max(0,     (int)std::round(x1));
    int by1 = std::max(0,     (int)std::round(y1));
    int bx2 = std::min(img_w, (int)std::round(x2));
    int by2 = std::min(img_h, (int)std::round(y2));
    if (bx2 > bx1 && by2 > by1)
        mask_orig(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1))
            .copyTo(result(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1)));

    cv::threshold(result, result, mask_threshold, 1.f, cv::THRESH_BINARY);
    return result;
}

void TRTYoloV8Seg::detect(
    const cv::Mat &mat,
    std::vector<types::BoxfWithSegMask> &detected_objects,
    float score_threshold, float iou_threshold,
    unsigned int topk, unsigned int /*nms_type*/)
{
    if (mat.empty()) return;

    int img_height = mat.rows;
    int img_width  = mat.cols;

    // 1. Letterbox preprocess
    cv::Mat mat_rs;
    YOLOv8SegScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, input_node_dims[2], input_node_dims[3], scale_params);

    // Convert BGR -> RGB, normalize
    cv::cvtColor(mat_rs, mat_rs, cv::COLOR_BGR2RGB);
    mat_rs.convertTo(mat_rs, CV_32F, scale_val, mean_val);

    // 2. Build input tensor and copy to device
    std::vector<float> input_data;
    trtcv::utils::transform::create_tensor(mat_rs, input_data, input_node_dims,
                                           trtcv::utils::transform::CHW);

    cudaMemcpyAsync(buffers[0], input_data.data(),
                    input_node_dims[0] * input_node_dims[1] *
                    input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // 3. Inference
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    if (!status)
    {
        std::cerr << "TRTYoloV8Seg: inference failed" << std::endl;
        return;
    }

    // 4. Retrieve output0 (detections)
    auto det_dims = output_node_dims[0]; // (1, 116, 8400)
    std::vector<float> det_output(det_dims[0] * det_dims[1] * det_dims[2]);
    cudaMemcpyAsync(det_output.data(), buffers[1],
                    det_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    // Retrieve output1 (prototypes)
    auto proto_dims = output_node_dims[1]; // (1, 32, 160, 160)
    std::vector<float> proto_output(proto_dims[0] * proto_dims[1] * proto_dims[2] * proto_dims[3]);
    cudaMemcpyAsync(proto_output.data(), buffers[2],
                    proto_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 5. Decode detections
    std::vector<types::Boxf> bbox_collection;
    std::vector<std::vector<float>> mask_coeffs_collection;
    generate_detections(scale_params, bbox_collection, mask_coeffs_collection,
                        det_output.data(), score_threshold, img_height, img_width);

    // 6. NMS
    std::vector<types::Boxf> nms_boxes;
    std::vector<std::vector<float>> nms_coeffs;
    this->nms(bbox_collection, nms_boxes, mask_coeffs_collection, nms_coeffs,
              iou_threshold, topk);

    // 7. Decode masks
    int proto_c = static_cast<int>(proto_dims[1]);
    int proto_h = static_cast<int>(proto_dims[2]);
    int proto_w = static_cast<int>(proto_dims[3]);

    detected_objects.clear();
    for (unsigned int i = 0; i < nms_boxes.size(); ++i)
    {
        types::BoxfWithSegMask obj;
        obj.box  = nms_boxes[i];
        obj.mask = decode_single_mask(
            nms_coeffs[i], proto_output.data(), proto_c, proto_h, proto_w,
            scale_params, img_height, img_width,
            nms_boxes[i].x1, nms_boxes[i].y1,
            nms_boxes[i].x2, nms_boxes[i].y2);
        obj.flag = true;
        detected_objects.push_back(obj);
    }
}
