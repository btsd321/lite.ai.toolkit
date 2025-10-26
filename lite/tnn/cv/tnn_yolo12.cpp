//
// Created by DefTruth on 2024/10/26.
//

#include "tnn_yolo12.h"
#include "lite/tnn/core/tnn_utils.h"
#include "lite/utils.h"

using tnncv::TNNYOLO12;

void TNNYOLO12::transform(const cv::Mat &mat_rs)
{
    // push input mat to input_mat
    input_mat = std::make_shared<TNN_NS::Mat>(device_type, TNN_NS::N8UC3,
                                              input_dims, (void *)mat_rs.data);
    if (!input_mat->GetData())
    {
#ifdef LITETNN_DEBUG
        std::cout << "input_mat == nullptr! transform failed\n";
#endif
    }
}

void TNNYOLO12::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                               int target_height, int target_width,
                               YOLO12ScaleParams &scale_params)
{
    if (mat.empty())
        return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                     cv::Scalar(114, 114, 114));
    // scale ratio (new / old) new_shape(h,w)
    float w_r = (float)target_width / (float)img_width;
    float h_r = (float)target_height / (float)img_height;
    float r = std::min(w_r, h_r);
    // compute padding
    int new_unpad_w = static_cast<int>((float)img_width * r);  // floor
    int new_unpad_h = static_cast<int>((float)img_height * r); // floor
    int pad_w = target_width - new_unpad_w;                    // >=0
    int pad_h = target_height - new_unpad_h;                   // >=0

    int dw = pad_w / 2;
    int dh = pad_h / 2;

    // resize with unscaling
    cv::Mat new_unpad_mat;
    cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
    new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

    // record scale params.
    scale_params.r = r;
    scale_params.dw = dw;
    scale_params.dh = dh;
    scale_params.new_unpad_w = new_unpad_w;
    scale_params.new_unpad_h = new_unpad_h;
    scale_params.flag = true;
}

void TNNYOLO12::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                       float score_threshold, float iou_threshold,
                       unsigned int topk, unsigned int nms_type)
{
    if (mat.empty())
        return;
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs;
    YOLO12ScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    // 1. make input tensor
    this->transform(mat_rs);
    // 2. inference scores & boxes.
    auto status = net->Forward();
    if (status != TNN_NS::TNN_OK)
    {
#ifdef LITETNN_DEBUG
        std::cout << "TNN forward failed!\n";
#endif
        return;
    }
    // 3. rescale & exclude.
    std::vector<types::Boxf> bbox_collection;
    std::shared_ptr<TNN_NS::Mat> pred;
    status = net->GetOutputMat(pred);
    if (status != TNN_NS::TNN_OK)
    {
#ifdef LITETNN_DEBUG
        std::cout << "TNN GetOutputMat failed!\n";
#endif
        return;
    }
    this->generate_bboxes(scale_params, bbox_collection, pred, score_threshold, img_height, img_width);
    // 4. hard|blend|offset nms with topk.
    this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void TNNYOLO12::generate_bboxes(const YOLO12ScaleParams &scale_params,
                                std::vector<types::Boxf> &bbox_collection,
                                std::shared_ptr<TNN_NS::Mat> &pred,
                                float score_threshold, int img_height, int img_width)
{
    auto pred_dims = pred->GetDims();
    const unsigned int num_anchors = pred_dims.at(1);     // n
    const unsigned int num_classes = pred_dims.at(2) - 4; // 80

    float r_ = scale_params.r;
    int dw_ = scale_params.dw;
    int dh_ = scale_params.dh;

    bbox_collection.clear();
    unsigned int count = 0;
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        float *offset_obj_cls_ptr = (float *)pred->GetData() + (i * (num_classes + 4));
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < score_threshold)
            continue; // filter first.

        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float tmp_conf = offset_obj_cls_ptr[j + 4];
            if (tmp_conf > cls_conf)
            {
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax

        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold)
            continue; // filter

        float cx = offset_obj_cls_ptr[0];
        float cy = offset_obj_cls_ptr[1];
        float w = offset_obj_cls_ptr[2];
        float h = offset_obj_cls_ptr[3];

        float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float)dh_) / r_;

        x1 = std::max(std::min(x1, (float)img_width - 1.f), 0.f);
        y1 = std::max(std::min(y1, (float)img_height - 1.f), 0.f);
        x2 = std::max(std::min(x2, (float)img_width - 1.f), 0.f);
        y2 = std::max(std::min(y2, (float)img_height - 1.f), 0.f);

        types::Boxf box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.score = conf;
        box.label = label;
        box.label_text = class_names[label];
        box.flag = true;
        bbox_collection.push_back(box);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

#if LITETNN_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void TNNYOLO12::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
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