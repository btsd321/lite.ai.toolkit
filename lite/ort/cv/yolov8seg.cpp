//
// Created for YOLOv8-Seg (instance segmentation) support
//

#include "yolov8seg.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::YOLOv8Seg;

Ort::Value YOLOv8Seg::transform(const cv::Mat &mat_rs)
{
    cv::Mat canvas;
    cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32FC3, scale_val, -mean_val * scale_val);
    // (1,3,height,width)
    return ortcv::utils::transform::create_tensor(
        canvas, input_node_dims, memory_info_handler,
        input_values_handler, ortcv::utils::transform::CHW);
}

void YOLOv8Seg::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                int target_height, int target_width,
                                YOLOv8SegScaleParams &scale_params)
{
    if (mat.empty()) return;
    int img_height = static_cast<int>(mat.rows);
    int img_width  = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(114, 114, 114));
    float w_r = (float)target_width  / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r   = std::min(w_r, h_r);

    int new_unpad_w = static_cast<int>((float)img_width  * r);
    int new_unpad_h = static_cast<int>((float)img_height * r);
    int pad_w = target_width  - new_unpad_w;
    int pad_h = target_height - new_unpad_h;
    int dw = pad_w / 2;
    int dh = pad_h / 2;

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

void YOLOv8Seg::generate_detections(
    const YOLOv8SegScaleParams &scale_params,
    std::vector<types::Boxf> &bbox_collection,
    std::vector<std::vector<float>> &mask_coeffs_collection,
    std::vector<Ort::Value> &output_tensors,
    float score_threshold, int img_height, int img_width)
{
    // output0: (1, n_anchors, 4+num_classes+32)
    Ort::Value &pred = output_tensors.at(0);
    auto pred_dims = output_node_dims.at(0);        // (1, n_anchors, 116)
    const unsigned int num_anchors = pred_dims.at(1);
    const unsigned int total_channels = pred_dims.at(2); // 4 + num_classes + 32
    const unsigned int num_classes = total_channels - 4 - num_mask_coeffs; // e.g. 80

    float r_  = scale_params.r;
    int   dw_ = scale_params.dw;
    int   dh_ = scale_params.dh;

    bbox_collection.clear();
    mask_coeffs_collection.clear();
    unsigned int count = 0;

    const float *data = pred.GetTensorData<float>();

    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        const float *anchor_ptr = data + i * total_channels;
        // cx, cy, w, h
        float cx = anchor_ptr[0];
        float cy = anchor_ptr[1];
        float w  = anchor_ptr[2];
        float h  = anchor_ptr[3];

        // Find best class score (indices 4 .. 4+num_classes-1)
        float max_cls_conf = -1.f;
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float s = anchor_ptr[4 + j];
            if (s > max_cls_conf)
            {
                max_cls_conf = s;
                label = j;
            }
        }

        if (max_cls_conf < score_threshold) continue;

        // Unscale to original image
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

        // Mask coefficients (indices 4+num_classes .. 4+num_classes+31)
        std::vector<float> coeffs(num_mask_coeffs);
        for (int k = 0; k < num_mask_coeffs; ++k)
            coeffs[k] = anchor_ptr[4 + num_classes + k];
        mask_coeffs_collection.push_back(coeffs);

        if (++count > max_nms) break;
    }

#if LITEORT_DEBUG
    std::cout << "YOLOv8Seg detected num_anchors: " << num_anchors << "\n";
    std::cout << "YOLOv8Seg generate_detections num: " << bbox_collection.size() << "\n";
#endif
}

void YOLOv8Seg::nms(
    std::vector<types::Boxf> &input,
    std::vector<types::Boxf> &output,
    std::vector<std::vector<float>> &coeffs_in,
    std::vector<std::vector<float>> &coeffs_out,
    float iou_threshold, unsigned int topk, unsigned int nms_type)
{
    // We use offset_nms (class-aware) by default; keep coeffs in sync
    if (input.empty()) return;

    // Sort by score descending (same logic as lite::utils::hard_nms internals)
    std::vector<unsigned int> indices(input.size());
    for (unsigned int i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&](unsigned int a, unsigned int b) { return input[a].score > input[b].score; });

    std::vector<types::Boxf> sorted_boxes;
    std::vector<std::vector<float>> sorted_coeffs;
    sorted_boxes.reserve(input.size());
    sorted_coeffs.reserve(input.size());
    for (auto idx : indices)
    {
        sorted_boxes.push_back(input[idx]);
        sorted_coeffs.push_back(coeffs_in[idx]);
    }

    // Hard NMS per class
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
            // Class-aware NMS
            if (nms_type != NMS::HARD && sorted_boxes[i].label != sorted_boxes[j].label)
                continue;
            float iou = sorted_boxes[i].iou_of(sorted_boxes[j]);
            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }
}

cv::Mat YOLOv8Seg::decode_single_mask(
    const std::vector<float> &coeffs,
    const float *proto_data,
    int proto_c, int proto_h, int proto_w,
    const YOLOv8SegScaleParams &scale_params,
    int img_h, int img_w,
    float x1, float y1, float x2, float y2)
{
    // 1. Compute mask in proto space: mask = coeffs @ proto.reshape(proto_c, proto_h*proto_w)
    int proto_area = proto_h * proto_w;
    std::vector<float> mask_vec(proto_area, 0.f);
    for (int k = 0; k < proto_c; ++k)
    {
        float ck = coeffs[k];
        const float *proto_k = proto_data + k * proto_area;
        for (int j = 0; j < proto_area; ++j)
            mask_vec[j] += ck * proto_k[j];
    }

    // 2. Apply sigmoid
    for (auto &v : mask_vec)
        v = 1.f / (1.f + std::exp(-v));

    // 3. Create mask_proto mat in proto space
    cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, mask_vec.data());

    // 4. Upsample to letterboxed input size
    int input_h = static_cast<int>(input_node_dims.at(2));
    int input_w = static_cast<int>(input_node_dims.at(3));
    cv::Mat mask_input;
    cv::resize(mask_proto, mask_input, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

    // 5. Crop out letterbox padding -> original image content region
    int x_start = scale_params.dw;
    int y_start = scale_params.dh;
    int crop_w  = scale_params.new_unpad_w;
    int crop_h  = scale_params.new_unpad_h;
    cv::Mat mask_crop = mask_input(cv::Rect(x_start, y_start, crop_w, crop_h)).clone();

    // 6. Resize to original image size
    cv::Mat mask_orig;
    cv::resize(mask_crop, mask_orig, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);

    // 7. Apply bounding box mask and threshold
    cv::Mat result = cv::Mat::zeros(img_h, img_w, CV_32FC1);
    int bx1 = std::max(0,     (int)std::round(x1));
    int by1 = std::max(0,     (int)std::round(y1));
    int bx2 = std::min(img_w, (int)std::round(x2));
    int by2 = std::min(img_h, (int)std::round(y2));
    if (bx2 > bx1 && by2 > by1)
    {
        cv::Mat roi_orig = mask_orig(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1));
        roi_orig.copyTo(result(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1)));
    }
    cv::threshold(result, result, mask_threshold, 1.f, cv::THRESH_BINARY);

    return result;
}

void YOLOv8Seg::detect(
    const cv::Mat &mat,
    std::vector<types::BoxfWithSegMask> &detected_objects,
    float score_threshold, float iou_threshold,
    unsigned int topk, unsigned int nms_type)
{
    if (mat.empty()) return;

    const int input_height = static_cast<int>(input_node_dims.at(2));
    const int input_width  = static_cast<int>(input_node_dims.at(3));
    int img_height = mat.rows;
    int img_width  = mat.cols;

    // 1. Letterbox resize
    cv::Mat mat_rs;
    YOLOv8SegScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    // 2. Build input tensor
    Ort::Value input_tensor = this->transform(mat_rs);

    // 3. Inference (2 outputs: detection + prototypes)
    auto output_tensors = ort_session->Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(), &input_tensor, 1,
        output_node_names.data(), num_outputs);

    // 4. Decode detections
    std::vector<types::Boxf> bbox_collection;
    std::vector<std::vector<float>> mask_coeffs_collection;
    this->generate_detections(scale_params, bbox_collection, mask_coeffs_collection,
                               output_tensors, score_threshold, img_height, img_width);

    // 5. NMS (keeping coefficients in sync)
    std::vector<types::Boxf> nms_boxes;
    std::vector<std::vector<float>> nms_coeffs;
    this->nms(bbox_collection, nms_boxes, mask_coeffs_collection, nms_coeffs,
              iou_threshold, topk, nms_type);

    // 6. Decode instance masks
    // output1: (1, proto_c, proto_h, proto_w)
    auto proto_dims = output_node_dims.at(1); // (1, 32, 160, 160)
    int proto_c = static_cast<int>(proto_dims.at(1));
    int proto_h = static_cast<int>(proto_dims.at(2));
    int proto_w = static_cast<int>(proto_dims.at(3));
    const float *proto_data = output_tensors.at(1).GetTensorData<float>();

    detected_objects.clear();
    for (unsigned int i = 0; i < nms_boxes.size(); ++i)
    {
        types::BoxfWithSegMask obj;
        obj.box  = nms_boxes[i];
        obj.mask = this->decode_single_mask(
            nms_coeffs[i], proto_data, proto_c, proto_h, proto_w,
            scale_params, img_height, img_width,
            nms_boxes[i].x1, nms_boxes[i].y1,
            nms_boxes[i].x2, nms_boxes[i].y2);
        obj.flag = true;
        detected_objects.push_back(obj);
    }
}
