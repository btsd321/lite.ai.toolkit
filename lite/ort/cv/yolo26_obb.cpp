//
// Created for YOLO26-OBB support
//

#include "yolo26_obb.h"
#include "lite/ort/core/ort_utils.h"
#include "lite/utils.h"

using ortcv::YOLO26OBB;

Ort::Value YOLO26OBB::transform(const cv::Mat &mat_rs)
{
    cv::Mat canvas;
    cv::cvtColor(mat_rs, canvas, cv::COLOR_BGR2RGB);
    canvas.convertTo(canvas, CV_32FC3, scale_val, mean_val);
    // (1,3,height,width)
    return ortcv::utils::transform::create_tensor(
        canvas, input_node_dims, memory_info_handler,
        input_values_handler, ortcv::utils::transform::CHW);
}

void YOLO26OBB::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                               int target_height, int target_width,
                               YOLO26OBBScaleParams &scale_params)
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

void YOLO26OBB::detect_model_type()
{
    if (output_node_dims.empty() || output_node_dims[0].size() < 3)
    {
        std::cerr << "Warning: Unable to determine model type, defaulting to end-to-end" << std::endl;
        is_end2end_ = true;
        return;
    }

    auto &first_output_dims = output_node_dims[0];
    
    // End-to-end model: [batch, num_detections, 7] e.g., [1, 300, 7]
    // Non-end-to-end model: [batch, num_channels, num_anchors] e.g., [1, 10, 8400]
    
    // If last dimension is 7 and middle dimension is relatively small (< 1000),
    // it's likely end-to-end with format [x1, y1, x2, y2, score, class_id, angle]
    if (first_output_dims[2] == 7 && first_output_dims[1] < 1000)
    {
        is_end2end_ = true;
#ifdef LITEORT_DEBUG
        std::cout << "Detected end-to-end model with output shape: ["
                  << first_output_dims[0] << ", " << first_output_dims[1] << ", " << first_output_dims[2] << "]" << std::endl;
#endif
    }
    // Otherwise, it's non-end-to-end with format [batch, channels, anchors]
    else
    {
        is_end2end_ = false;
#ifdef LITEORT_DEBUG
        std::cout << "Detected non-end-to-end model with output shape: ["
                  << first_output_dims[0] << ", " << first_output_dims[1] << ", " << first_output_dims[2] << "]" << std::endl;
#endif
    }
}

void YOLO26OBB::generate_bboxes_obb(const YOLO26OBBScaleParams &scale_params,
                                    std::vector<types::BoxfWithAngle> &bbox_collection,
                                    std::vector<Ort::Value> &output_tensors,
                                    float score_threshold,
                                    int img_height,
                                    int img_width)
{
    Ort::Value &output = output_tensors.at(0); // [1, 300, 7]
    auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    
    const unsigned int num_detections = output_dims[1]; // 300
    const unsigned int detection_dim = output_dims[2];  // 7
    
    if (detection_dim != 7)
    {
        std::cerr << "Warning: Expected 7 values per detection, got " << detection_dim << std::endl;
        return;
    }
    
    const float *output_ptr = output.GetTensorData<float>();
    bbox_collection.clear();

    for (unsigned int i = 0; i < num_detections; ++i)
    {
        const float *detection = output_ptr + i * detection_dim;
        
        // Parse detection: [x1, y1, x2, y2, score, class_id, angle]
        float x1 = detection[0];
        float y1 = detection[1];
        float x2 = detection[2];
        float y2 = detection[3];
        float score = detection[4];
        float class_id_f = detection[5];
        float angle = detection[6];
        
        // Filter by score threshold
        if (score < score_threshold)
            continue;
        
        unsigned int label = static_cast<unsigned int>(class_id_f);
        
        // Unscale coordinates back to original image size
        // The model outputs are in input image space (640x640)
        // Need to map back to original image space
        if (scale_params.flag)
        {
            x1 = (x1 - scale_params.dw) / scale_params.r;
            y1 = (y1 - scale_params.dh) / scale_params.r;
            x2 = (x2 - scale_params.dw) / scale_params.r;
            y2 = (y2 - scale_params.dh) / scale_params.r;
        }
        
        // Clamp to image boundaries
        x1 = std::max(0.f, std::min(x1, (float)img_width - 1.f));
        y1 = std::max(0.f, std::min(y1, (float)img_height - 1.f));
        x2 = std::max(0.f, std::min(x2, (float)img_width - 1.f));
        y2 = std::max(0.f, std::min(y2, (float)img_height - 1.f));
        
        // Calculate center and dimensions for rotated box
        float cx = (x1 + x2) / 2.f;
        float cy = (y1 + y2) / 2.f;
        float width = x2 - x1;
        float height = y2 - y1;
        
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

#ifdef LITEORT_DEBUG
    std::cout << "YOLO26OBB detected " << bbox_collection.size() 
              << " boxes after threshold filtering.\n";
#endif
}

void YOLO26OBB::generate_bboxes_obb_non_end2end(const YOLO26OBBScaleParams &scale_params,
                                                std::vector<types::BoxfWithAngle> &bbox_collection,
                                                std::vector<Ort::Value> &output_tensors,
                                                float score_threshold,
                                                int img_height,
                                                int img_width)
{
    Ort::Value &output = output_tensors.at(0);
    auto output_dims = output.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    
    const unsigned int num_anchors = output_dims[2]; // e.g., 8400
    const unsigned int num_classes = output_dims[1] - 5; // total_channels - (4 bbox + 1 angle)
    
    // Model output format: [cx, cy, w, h, cls0, cls1, ..., clsN-1, angle]
    // ch0-3: bbox coordinates, ch4 to ch(4+num_classes-1): class scores, last channel: angle
    
    const float *output_ptr = output.GetTensorData<float>();
    
    float x_factor = float(img_width) / input_node_dims.at(3);
    float y_factor = float(img_height) / input_node_dims.at(2);

    bbox_collection.clear();
    unsigned int count = 0;
    
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        // Extract class scores
        float max_cls_conf = -FLT_MAX;
        unsigned int label = 0;
        
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            float score = output_ptr[(4 + j) * num_anchors + i];
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
        float cx = output_ptr[0 * num_anchors + i];
        float cy = output_ptr[1 * num_anchors + i];
        float w = output_ptr[2 * num_anchors + i];
        float h = output_ptr[3 * num_anchors + i];

        // Extract angle from last channel
        float angle = output_ptr[(4 + num_classes) * num_anchors + i];

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

#ifdef LITEORT_DEBUG
    std::cout << "YOLO26OBB (non-end2end) detected " << bbox_collection.size() 
              << " boxes after threshold filtering.\n";
#endif
}

float YOLO26OBB::compute_obb_iou(const types::BoxfWithAngle &box1,
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

void YOLO26OBB::nms_obb(std::vector<types::BoxfWithAngle> &input,
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

void YOLO26OBB::detect(const cv::Mat &mat, 
                       std::vector<types::BoxfWithAngle> &detected_boxes,
                       float score_threshold, 
                       float iou_threshold,
                       unsigned int topk)
{
    if (mat.empty())
        return;
    
    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // resize & unscale
    cv::Mat mat_rs;
    YOLO26OBBScaleParams scale_params;
    this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

    // 1. make input tensor
    Ort::Value input_tensor = this->transform(mat_rs);
    
    // 2. inference
    auto output_tensors = ort_session->Run(
        Ort::RunOptions{nullptr}, 
        input_node_names.data(),
        &input_tensor, 
        1, 
        output_node_names.data(), 
        num_outputs
    );

    // 3. Post-processing based on model type
    detected_boxes.clear();
    
    if (is_end2end_)
    {
        // End-to-end model: NMS already applied in ONNX model
        this->generate_bboxes_obb(scale_params, detected_boxes, output_tensors, 
                                  score_threshold, img_height, img_width);
        
        // 4. Apply topk limit if needed
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
        this->generate_bboxes_obb_non_end2end(scale_params, bbox_collection, output_tensors,
                                               score_threshold, img_height, img_width);
        
        // 4. Apply NMS
        this->nms_obb(bbox_collection, detected_boxes, iou_threshold, topk);
    }

#ifdef LITEORT_DEBUG
    std::cout << "YOLO26OBB final detections: " << detected_boxes.size() << "\n";
#endif
}
