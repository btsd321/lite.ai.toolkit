//
// Created for YOLO26-Seg TensorRT support
//

#include "trt_yolo26seg.h"
#include <opencv2/opencv.hpp>

using trtcv::TRTYolo26Seg;

void TRTYolo26Seg::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                  int target_height, int target_width,
                                  YOLO26SegScaleParams &scale_params)
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

void TRTYolo26Seg::generate_detections(
    const YOLO26SegScaleParams &scale_params,
    std::vector<types::Boxf> &bbox_collection,
    std::vector<std::vector<float>> &mask_coeffs_collection,
    const float *det_output,
    float score_threshold, int img_height, int img_width)
{
    // YOLO26 end-to-end output0: 检测优先 (1, N, detection_dim)
    // detection_dim = num_box_fields(6) + num_mask_coeffs(32) = 38
    // 每条检测: [x1, y1, x2, y2, score, class_id, coeff0..coeff31]
    auto pred_dims = output_node_dims[0]; // (1, N, 38)
    const unsigned int num_detections = static_cast<unsigned int>(pred_dims[1]);
    const unsigned int detection_dim  = static_cast<unsigned int>(pred_dims[2]);

    float r_  = scale_params.r;
    int   dw_ = scale_params.dw;
    int   dh_ = scale_params.dh;

    bbox_collection.clear();
    mask_coeffs_collection.clear();

    for (unsigned int i = 0; i < num_detections; ++i)
    {
        const float *det = det_output + i * detection_dim;

        // YOLO26-seg 端到端导出为 xyxy 角点格式（letterbox 输入空间）
        float x1_in  = det[0];
        float y1_in  = det[1];
        float x2_in  = det[2];
        float y2_in  = det[3];
        float score  = det[4];
        float cls_id = det[5];

        // 引擎内已做 NMS，这里只按置信度阈值过滤
        if (score < score_threshold) continue;

        // 去 letterbox padding，映射回原图坐标
        float x1 = (x1_in - (float)dw_) / r_;
        float y1 = (y1_in - (float)dh_) / r_;
        float x2 = (x2_in - (float)dw_) / r_;
        float y2 = (y2_in - (float)dh_) / r_;

        x1 = std::max(std::min(x1, (float)img_width  - 1.f), 0.f);
        y1 = std::max(std::min(y1, (float)img_height - 1.f), 0.f);
        x2 = std::max(std::min(x2, (float)img_width  - 1.f), 0.f);
        y2 = std::max(std::min(y2, (float)img_height - 1.f), 0.f);

        unsigned int label = static_cast<unsigned int>(cls_id);

        types::Boxf box;
        box.x1         = x1;
        box.y1         = y1;
        box.x2         = x2;
        box.y2         = y2;
        box.score      = score;
        box.label      = label;
        box.label_text = (use_custom_class_names && label < custom_class_names.size())
                             ? custom_class_names[label].c_str()
                             : (label < 80 ? class_names[label] : "unknown");
        box.flag       = true;
        bbox_collection.push_back(box);

        std::vector<float> coeffs(num_mask_coeffs);
        for (int k = 0; k < num_mask_coeffs; ++k)
            coeffs[k] = det[num_box_fields + k];
        mask_coeffs_collection.push_back(coeffs);
    }

#if LITETRT_DEBUG
    std::cout << "TRTYolo26Seg generate_detections num: " << bbox_collection.size() << "\n";
#endif
}

cv::Mat TRTYolo26Seg::decode_single_mask(
    const std::vector<float> &coeffs,
    const float *proto_data,
    int proto_c, int proto_h, int proto_w,
    const YOLO26SegScaleParams &scale_params,
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

void TRTYolo26Seg::detect(
    const cv::Mat &mat,
    std::vector<types::BoxfWithSegMask> &detected_objects,
    float score_threshold, float /*iou_threshold*/,
    unsigned int topk, unsigned int /*nms_type*/)
{
    if (mat.empty()) return;

    int img_height = mat.rows;
    int img_width  = mat.cols;

    // 1. Letterbox preprocess
    cv::Mat mat_rs;
    YOLO26SegScaleParams scale_params;
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
        std::cerr << "TRTYolo26Seg: inference failed" << std::endl;
        return;
    }

    // 4. Retrieve output0 (detections)
    auto det_dims = output_node_dims[0]; // (1, N, 38)
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

    // 5. Decode detections (NMS-free, 仅置信度过滤)
    std::vector<types::Boxf> bbox_collection;
    std::vector<std::vector<float>> mask_coeffs_collection;
    generate_detections(scale_params, bbox_collection, mask_coeffs_collection,
                        det_output.data(), score_threshold, img_height, img_width);

    // 6. topk 限制（按 score 降序）
    if (bbox_collection.size() > topk)
    {
        std::vector<unsigned int> idx(bbox_collection.size());
        for (unsigned int i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](unsigned int a, unsigned int b)
                  { return bbox_collection[a].score > bbox_collection[b].score; });
        std::vector<types::Boxf> tmp_boxes;
        std::vector<std::vector<float>> tmp_coeffs;
        for (unsigned int t = 0; t < topk; ++t)
        {
            tmp_boxes.push_back(bbox_collection[idx[t]]);
            tmp_coeffs.push_back(mask_coeffs_collection[idx[t]]);
        }
        bbox_collection.swap(tmp_boxes);
        mask_coeffs_collection.swap(tmp_coeffs);
    }

    // 7. Decode masks
    int proto_c = static_cast<int>(proto_dims[1]);
    int proto_h = static_cast<int>(proto_dims[2]);
    int proto_w = static_cast<int>(proto_dims[3]);

    detected_objects.clear();
    for (unsigned int i = 0; i < bbox_collection.size(); ++i)
    {
        types::BoxfWithSegMask obj;
        obj.box  = bbox_collection[i];
        obj.mask = decode_single_mask(
            mask_coeffs_collection[i], proto_output.data(), proto_c, proto_h, proto_w,
            scale_params, img_height, img_width,
            bbox_collection[i].x1, bbox_collection[i].y1,
            bbox_collection[i].x2, bbox_collection[i].y2);
        obj.flag = true;
        detected_objects.push_back(obj);
    }
}

// ==================== GPU Path ====================

void TRTYolo26Seg::ensure_gpu_components()
{
    if (!gpu_preprocessor_)
    {
        int model_h = static_cast<int>(input_node_dims[2]);
        int model_w = static_cast<int>(input_node_dims[3]);
        gpu_preprocessor_ = std::make_unique<trtgpu::GpuPreprocessor>(model_h, model_w, 114.f);
    }
    if (!gpu_mask_decoder_)
    {
        gpu_mask_decoder_ = std::make_unique<trtgpu::GpuMaskDecoder>(num_mask_coeffs, mask_threshold);
    }
}

void TRTYolo26Seg::detect_gpu(
    const cv::Mat &mat,
    std::vector<GpuSegDetection> &detected_objects,
    float score_threshold, float iou_threshold,
    unsigned int topk)
{
    if (mat.empty()) return;
    ensure_gpu_components();

    trtgpu::ScaleParams sp{};
    gpu_preprocessor_->preprocess(mat, static_cast<float*>(buffers[0]),
                                  sp, stream, /*bgr2rgb=*/true);

    detect_gpu_impl(sp, mat.rows, mat.cols,
                    detected_objects, score_threshold, iou_threshold, topk);
}

void TRTYolo26Seg::detect_gpu(
    const cv::cuda::GpuMat &gpu_mat,
    int img_height, int img_width,
    std::vector<GpuSegDetection> &detected_objects,
    float score_threshold, float iou_threshold,
    unsigned int topk)
{
    if (gpu_mat.empty()) return;
    ensure_gpu_components();

    trtgpu::ScaleParams sp{};
    gpu_preprocessor_->preprocess(gpu_mat, static_cast<float*>(buffers[0]),
                                  sp, stream, /*bgr2rgb=*/true);

    detect_gpu_impl(sp, img_height, img_width,
                    detected_objects, score_threshold, iou_threshold, topk);
}

void TRTYolo26Seg::detect_gpu_impl(
    const trtgpu::ScaleParams &sp,
    int img_height, int img_width,
    std::vector<GpuSegDetection> &detected_objects,
    float score_threshold, float /*iou_threshold*/,
    unsigned int topk)
{
    // 1. TRT 推理
    cudaStreamSynchronize(stream);
    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    if (!status)
    {
        std::cerr << "TRTYolo26Seg: GPU inference failed" << std::endl;
        return;
    }

    // 2. 只回传 det 输出（proto 留在 GPU 上）
    auto det_dims = output_node_dims[0]; // (1, N, 38)
    std::vector<float> det_output(det_dims[0] * det_dims[1] * det_dims[2]);
    cudaMemcpyAsync(det_output.data(), buffers[1],
                    det_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. ScaleParams → YOLO26SegScaleParams 转换
    YOLO26SegScaleParams yolo_sp{};
    yolo_sp.r = sp.ratio;
    yolo_sp.dw = sp.pad_left;
    yolo_sp.dh = sp.pad_top;
    yolo_sp.new_unpad_w = sp.resized_w;
    yolo_sp.new_unpad_h = sp.resized_h;
    yolo_sp.flag = true;

    // 4. CPU 解码（NMS-free，仅置信度过滤）
    std::vector<types::Boxf> bbox_collection;
    std::vector<std::vector<float>> mask_coeffs_collection;
    generate_detections(yolo_sp, bbox_collection, mask_coeffs_collection,
                        det_output.data(), score_threshold, img_height, img_width);

    // 5. topk 限制（按 score 降序）
    if (bbox_collection.size() > topk)
    {
        std::vector<unsigned int> idx(bbox_collection.size());
        for (unsigned int i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](unsigned int a, unsigned int b)
                  { return bbox_collection[a].score > bbox_collection[b].score; });
        std::vector<types::Boxf> tmp_boxes;
        std::vector<std::vector<float>> tmp_coeffs;
        for (unsigned int t = 0; t < topk; ++t)
        {
            tmp_boxes.push_back(bbox_collection[idx[t]]);
            tmp_coeffs.push_back(mask_coeffs_collection[idx[t]]);
        }
        bbox_collection.swap(tmp_boxes);
        mask_coeffs_collection.swap(tmp_coeffs);
    }

    if (bbox_collection.empty())
    {
        detected_objects.clear();
        return;
    }

    // 6. 打包为 DetectionInfo
    std::vector<trtgpu::DetectionInfo> det_infos(bbox_collection.size());
    for (size_t i = 0; i < bbox_collection.size(); ++i)
    {
        det_infos[i].x1 = bbox_collection[i].x1;
        det_infos[i].y1 = bbox_collection[i].y1;
        det_infos[i].x2 = bbox_collection[i].x2;
        det_infos[i].y2 = bbox_collection[i].y2;
        det_infos[i].score = bbox_collection[i].score;
        det_infos[i].label = bbox_collection[i].label;
        det_infos[i].mask_coeffs = mask_coeffs_collection[i];
    }

    // 7. GPU 批量 mask 解码（proto 零拷贝）
    auto proto_dims = output_node_dims[1]; // (1, 32, 160, 160)
    int proto_h = static_cast<int>(proto_dims[2]);
    int proto_w = static_cast<int>(proto_dims[3]);
    int input_h = static_cast<int>(input_node_dims[2]);
    int input_w = static_cast<int>(input_node_dims[3]);

    auto gpu_masks = gpu_mask_decoder_->decode(
        static_cast<const float*>(buffers[2]),
        proto_h, proto_w,
        det_infos,
        input_h, input_w,
        sp.pad_left, sp.pad_top,
        sp.resized_w, sp.resized_h,
        img_height, img_width,
        stream);

    // 8. 组装结果
    detected_objects.clear();
    detected_objects.resize(bbox_collection.size());
    for (size_t i = 0; i < bbox_collection.size(); ++i)
    {
        detected_objects[i].box = bbox_collection[i];
        detected_objects[i].gpu_mask = std::move(gpu_masks[i]);
        detected_objects[i].flag = true;
    }
}
