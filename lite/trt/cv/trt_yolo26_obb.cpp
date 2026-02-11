//
// Created for YOLO26-OBB TensorRT support
//

#include "trt_yolo26_obb.h"
#include <opencv2/opencv.hpp>

using trtcv::TRTYOLO26OBB;

TRTYOLO26OBB::TRTYOLO26OBB(const std::string &_trt_model_path, unsigned int _num_threads)
    : BasicTRTHandler(_trt_model_path, _num_threads)
{
    init_default_class_names();
    detect_model_type();
}

TRTYOLO26OBB::TRTYOLO26OBB(const std::string &_trt_model_path, ImageFormat _input_format,
                           unsigned int _num_threads)
    : BasicTRTHandler(_trt_model_path, _num_threads), input_format_(_input_format)
{
    init_default_class_names();
    detect_model_type();
}

void TRTYOLO26OBB::preprocess(const cv::Mat &input_image, cv::Mat &output_mat)
{
    output_mat = input_image.clone();
    
    // Convert color space based on input format
    if (input_format_ == ImageFormat::BGR)
    {
        cv::cvtColor(output_mat, output_mat, cv::COLOR_BGR2RGB);
    }
    // If RGB, no conversion needed

    // Resize image to model input size
    cv::resize(output_mat, output_mat, 
               cv::Size(input_node_dims[3], input_node_dims[2]), 
               0, 0, cv::INTER_LINEAR);

    // Normalize image: convert to float32 and scale to [0, 1]
    output_mat.convertTo(output_mat, CV_32F, scale_val, mean_val);
}

void TRTYOLO26OBB::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                  int target_height, int target_width,
                                  YOLO26OBBScaleParams &scale_params)
{
    if (mat.empty())
        return;
    
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(114, 114, 114));
    
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

void TRTYOLO26OBB::detect_model_type()
{
    if (output_node_dims.empty() || output_node_dims[0].empty() || output_node_dims[0].size() < 3)
    {
        std::cerr << "Warning: Unable to determine model type, defaulting to end-to-end" << std::endl;
        is_end2end_ = true;
        return;
    }

    auto pred_dims = output_node_dims[0];
    
    // End-to-end model: [batch, num_detections, 7] e.g., [1, 300, 7]
    // Non-end-to-end model: [batch, num_channels, num_anchors] e.g., [1, 10, 8400]
    
    // If last dimension is 7 and middle dimension is relatively small (< 1000),
    // it's likely end-to-end with format [x1, y1, x2, y2, score, class_id, angle]
    if (pred_dims[2] == 7 && pred_dims[1] < 1000)
    {
        is_end2end_ = true;
#if LITETRT_DEBUG
        std::cout << "Detected end-to-end model with output shape: ["
                  << pred_dims[0] << ", " << pred_dims[1] << ", " << pred_dims[2] << "]" << std::endl;
#endif
    }
    // Otherwise, it's non-end-to-end with format [batch, channels, anchors]
    else
    {
        is_end2end_ = false;
#if LITETRT_DEBUG
        std::cout << "Detected non-end-to-end model with output shape: ["
                  << pred_dims[0] << ", " << pred_dims[1] << ", " << pred_dims[2] << "]" << std::endl;
#endif
    }
}

void TRTYOLO26OBB::generate_bboxes_obb(const YOLO26OBBScaleParams &scale_params,
                                       std::vector<types::BoxfWithAngle> &bbox_collection,
                                       float *output,
                                       float score_threshold,
                                       int img_height,
                                       int img_width)
{
    if (output_node_dims.empty() || output_node_dims[0].empty())
    {
        std::cerr << "Error: Invalid output dimensions" << std::endl;
        return;
    }

    auto pred_dims = output_node_dims[0];
    
    // Expected output format: [batch, num_detections, 7]
    // where 7 = [x1, y1, x2, y2, score, class_id, angle]
    const unsigned int num_detections = pred_dims[1]; // e.g., 300
    const unsigned int detection_dim = pred_dims[2];  // should be 7
    
    if (detection_dim != 7)
    {
        std::cerr << "Warning: Expected 7 values per detection, got " << detection_dim << std::endl;
        return;
    }

    bbox_collection.clear();

    for (unsigned int i = 0; i < num_detections; ++i)
    {
        const float *detection = output + i * detection_dim;
        
        // Parse detection: [cx, cy, w, h, score, class_id, angle]
        float cx = detection[0];
        float cy = detection[1];
        float width = detection[2];
        float height = detection[3];
        float score = detection[4];
        float class_id_f = detection[5];
        float angle = detection[6];
        
        // Filter by score threshold
        if (score < score_threshold)
            continue;
        
        unsigned int label = static_cast<unsigned int>(class_id_f);
        
        // Unscale coordinates back to original image size
        // The model outputs are in input image space (640x640) with center point format
        if (scale_params.flag)
        {
            cx = (cx - scale_params.dw) / scale_params.r;
            cy = (cy - scale_params.dh) / scale_params.r;
            width = width / scale_params.r;
            height = height / scale_params.r;
        }
        
        // Calculate corner coordinates from center
        float x1 = cx - width / 2.f;
        float y1 = cy - height / 2.f;
        float x2 = cx + width / 2.f;
        float y2 = cy + height / 2.f;
        
        // Clamp to image boundaries
        x1 = std::max(0.f, std::min(x1, (float)img_width - 1.f));
        y1 = std::max(0.f, std::min(y1, (float)img_height - 1.f));
        x2 = std::max(0.f, std::min(x2, (float)img_width - 1.f));
        y2 = std::max(0.f, std::min(y2, (float)img_height - 1.f));
        
        types::BoxfWithAngle box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.cx = cx;
        box.cy = cy;
        box.width = width;
        box.height = height;
        box.angle = angle; // radians
        box.score = score;
        box.label = label;
        box.label_text = get_class_name(label);
        box.flag = true;
        
        bbox_collection.push_back(box);
    }

#if LITETRT_DEBUG
    std::cout << "TRTYOLO26OBB detected " << bbox_collection.size() 
              << " boxes after threshold filtering.\n";
#endif
}

void TRTYOLO26OBB::generate_bboxes_obb_non_end2end(const YOLO26OBBScaleParams &scale_params,
                                                   std::vector<types::BoxfWithAngle> &bbox_collection,
                                                   float *output,
                                                   float score_threshold,
                                                   int img_height,
                                                   int img_width)
{
    if (output_node_dims.empty() || output_node_dims[0].empty())
    {
        std::cerr << "Error: Invalid output dimensions" << std::endl;
        return;
    }

    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims[2]; // e.g., 8400
    const unsigned int num_classes = pred_dims[1] - 5; // total_channels - (4 bbox + 1 angle)
    
    // Model output format: [cx, cy, w, h, cls0, cls1, ..., clsN-1, angle]
    // ch0-3: bbox coordinates, ch4 to ch(4+num_classes-1): class scores, last channel: angle

    float x_factor = float(img_width) / input_node_dims[3];
    float y_factor = float(img_height) / input_node_dims[2];

    bbox_collection.clear();
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        // Extract class scores
        float max_cls_conf = -FLT_MAX;
        unsigned int label = 0;
        
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float score = output[(4 + j) * num_anchors + i];
            if (score > max_cls_conf)
            {
                max_cls_conf = score;
                label = j;
            }
        }

        float conf = max_cls_conf;
        
        if (conf < score_threshold)
            continue;

        // Extract bbox (cx, cy, w, h)
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];

        // Extract angle from last channel
        float angle = output[(4 + num_classes) * num_anchors + i];

        // Calculate x1, y1 first, then scale
        float x1 = (cx - w / 2.f) * x_factor;
        float y1 = (cy - h / 2.f) * y_factor;

        w = w * x_factor;
        h = h * y_factor;

        float x2 = x1 + w;
        float y2 = y1 + h;

        // Scale center for rotated box
        cx = cx * x_factor;
        cy = cy * y_factor;

        types::BoxfWithAngle box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)img_width - 1.f);
        box.y2 = std::min(y2, (float)img_height - 1.f);
        box.cx = cx;
        box.cy = cy;
        box.width = w;
        box.height = h;
        box.angle = angle; // radians
        box.score = conf;
        box.label = label;
        box.label_text = get_class_name(label);
        box.flag = true;

        bbox_collection.push_back(box);

        count += 1;
        if (count > max_nms)
            break;
    }

#if LITETRT_DEBUG
    std::cout << "TRTYOLO26OBB (non-end2end) detected " << bbox_collection.size() 
              << " boxes after threshold filtering.\n";
#endif
}

float TRTYOLO26OBB::compute_obb_iou(const types::BoxfWithAngle &box1,
                                    const types::BoxfWithAngle &box2)
{
    // Create RotatedRect for OpenCV
    cv::RotatedRect rbox1(
        cv::Point2f(box1.cx, box1.cy),
        cv::Size2f(box1.width, box1.height),
        box1.angle * 180.f / CV_PI // Convert radians to degrees
    );

    cv::RotatedRect rbox2(
        cv::Point2f(box2.cx, box2.cy),
        cv::Size2f(box2.width, box2.height),
        box2.angle * 180.f / CV_PI);

    // Compute intersection
    std::vector<cv::Point2f> intersection;
    int result = cv::rotatedRectangleIntersection(rbox1, rbox2, intersection);

    if (result == cv::INTERSECT_NONE)
    {
        return 0.0f;
    }

    float intersection_area = 0.0f;
    if (!intersection.empty())
    {
        intersection_area = cv::contourArea(intersection);
    }

    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection_area;

    return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
}

void TRTYOLO26OBB::nms_obb(std::vector<types::BoxfWithAngle> &input,
                           std::vector<types::BoxfWithAngle> &output,
                           float iou_threshold,
                           unsigned int topk)
{
    if (input.empty())
        return;

    // Sort by confidence score (descending)
    std::sort(input.begin(), input.end(),
              [](const types::BoxfWithAngle &a, const types::BoxfWithAngle &b)
              {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(input.size(), false);

    for (size_t i = 0; i < input.size(); ++i)
    {
        if (suppressed[i])
            continue;

        output.push_back(input[i]);
        if (output.size() >= topk)
            break;

        for (size_t j = i + 1; j < input.size(); ++j)
        {
            if (suppressed[j])
                continue;

            // Only NMS for same class
            if (input[i].label != input[j].label)
                continue;

            float iou = compute_obb_iou(input[i], input[j]);
            
            if (iou > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }
}

void TRTYOLO26OBB::detect(const cv::Mat &mat,
                          std::vector<types::BoxfWithAngle> &detected_boxes,
                          float score_threshold,
                          float iou_threshold,
                          unsigned int topk)
{
    if (mat.empty())
        return;

    if (output_node_dims.empty() || output_node_dims[0].empty())
    {
        std::cerr << "Error: TensorRT engine not properly initialized. Output dimensions are empty." << std::endl;
        return;
    }

    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);
    
    const int input_height = input_node_dims[2];
    const int input_width = input_node_dims[3];

    // Resize & unscale
    cv::Mat mat_rs;
    YOLO26OBBScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    // Preprocess
    cv::Mat preprocessed;
    preprocess(mat_rs, preprocessed);

    // 1. Create input tensor
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(preprocessed, input, input_node_dims, 
                                           trtcv::utils::transform::CHW);

    // 2. Copy to device and infer
    cudaMemcpyAsync(buffers[0], input.data(),
                    input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    bool status = trt_context->enqueueV3(stream);
    
    if (!status)
    {
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);

    // 3. Get output
    auto pred_dims = output_node_dims[0];
    const unsigned int output_size = pred_dims[0] * pred_dims[1] * pred_dims[2];
    std::vector<float> output(output_size);

    cudaMemcpyAsync(output.data(), buffers[1],
                    output_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. Post-processing based on model type
    detected_boxes.clear();
    
    if (is_end2end_)
    {
        // End-to-end model: NMS already applied in engine
        this->generate_bboxes_obb(scale_params, detected_boxes, output.data(), 
                                  score_threshold, img_height, img_width);
        
        // 5. Apply topk limit if needed
        if (detected_boxes.size() > topk)
        {
            // Sort by confidence score
            std::sort(detected_boxes.begin(), detected_boxes.end(),
                      [](const types::BoxfWithAngle &a, const types::BoxfWithAngle &b)
                      {
                          return a.score > b.score;
                      });
            detected_boxes.resize(topk);
        }
    }
    else
    {
        // Non-end-to-end model: need to apply NMS manually
        std::vector<types::BoxfWithAngle> bbox_collection;
        this->generate_bboxes_obb_non_end2end(scale_params, bbox_collection, output.data(),
                                               score_threshold, img_height, img_width);
        
        // 5. Apply NMS
        this->nms_obb(bbox_collection, detected_boxes, iou_threshold, topk);
    }

#if LITETRT_DEBUG
    std::cout << "TRTYOLO26OBB final detections: " << detected_boxes.size() << "\n";
#endif
}
