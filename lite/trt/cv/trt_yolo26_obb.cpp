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
}

TRTYOLO26OBB::TRTYOLO26OBB(const std::string &_trt_model_path, ImageFormat _input_format,
                           unsigned int _num_threads)
    : BasicTRTHandler(_trt_model_path, _num_threads), input_format_(_input_format)
{
    init_default_class_names();
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

#if LITETRT_DEBUG
    std::cout << "TRTYOLO26OBB detected " << bbox_collection.size() 
              << " boxes after threshold filtering.\n";
#endif
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

    // 4. Post-processing (NMS already applied in engine)
    detected_boxes.clear();
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

#if LITETRT_DEBUG
    std::cout << "TRTYOLO26OBB final detections: " << detected_boxes.size() << "\n";
#endif
}
