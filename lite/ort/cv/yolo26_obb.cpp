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

    // 3. generate bounding boxes (NMS already applied in ONNX model)
    detected_boxes.clear();
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

#ifdef LITEORT_DEBUG
    std::cout << "YOLO26OBB final detections: " << detected_boxes.size() << "\n";
#endif
}
