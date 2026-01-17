//
// Created for YOLOv8-OBB support
//

#include "trt_yolov8_obb.h"
#include <opencv2/opencv.hpp>

using trtcv::TRTYoloV8OBB;

void TRTYoloV8OBB::preprocess(cv::Mat &input_image)
{
    // Convert color space from BGR to RGB
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    // Resize image
    cv::resize(input_image, input_image, cv::Size(input_node_dims[2], input_node_dims[3]), 0, 0, cv::INTER_LINEAR);

    // Normalize image
    input_image.convertTo(input_image, CV_32F, scale_val, mean_val);
}

void TRTYoloV8OBB::generate_bboxes_obb(
    std::vector<types::BoxfWithAngle> &bbox_collection,
    float *output,
    float score_threshold,
    int img_height,
    int img_width)
{

    auto pred_dims = output_node_dims[0];
    const unsigned int num_anchors = pred_dims[2];     // e.g., 8400
    const unsigned int num_classes = pred_dims[1] - 5; // bbox(4) + angle(1) + classes

    float x_factor = float(img_width) / input_node_dims[3];
    float y_factor = float(img_height) / input_node_dims[2];

    bbox_collection.clear();
    unsigned int count = 0;

    // YOLOv8-OBB output format: [cx, cy, w, h, angle, class0, class1, ...]
    // Transpose layout: [4+1+nc, num_anchors] -> iterate over num_anchors
    for (unsigned int i = 0; i < num_anchors; ++i)
    {
        // Extract class scores
        std::vector<float> class_scores(num_classes);
        for (unsigned int j = 0; j < num_classes; ++j)
        {
            class_scores[j] = output[(5 + j) * num_anchors + i];
        }

        auto max_it = std::max_element(class_scores.begin(), class_scores.end());
        float max_cls_conf = *max_it;
        unsigned int label = std::distance(class_scores.begin(), max_it);

        float conf = max_cls_conf;
        if (conf < score_threshold)
            continue;

        // Extract bbox (cx, cy, w, h)
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w = output[2 * num_anchors + i];
        float h = output[3 * num_anchors + i];

        // Extract angle (in radians)
        float angle = output[4 * num_anchors + i];

        // Scale to original image size
        cx = cx * x_factor;
        cy = cy * y_factor;
        w = w * x_factor;
        h = h * y_factor;

        // Convert to x1, y1, x2, y2 for axis-aligned bbox approximation
        float x1 = cx - w / 2.f;
        float y1 = cy - h / 2.f;
        float x2 = cx + w / 2.f;
        float y2 = cy + h / 2.f;

        types::BoxfWithAngle box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float)img_width - 1.f);
        box.y2 = std::min(y2, (float)img_height - 1.f);
        box.cx = cx;
        box.cy = cy;
        box.width = w;
        box.height = h;
        box.angle = angle; // radians, range: [-pi/4, 3*pi/4)
        box.score = conf;
        box.label = label;
        box.label_text = class_names[label];
        box.flag = true;

        bbox_collection.push_back(box);

        count += 1;
        if (count > max_nms)
            break;
    }

#if LITETRT_DEBUG
    std::cout << "detected num_anchors: " << num_anchors << "\n";
    std::cout << "generate_bboxes_obb num: " << bbox_collection.size() << "\n";
#endif
}

float TRTYoloV8OBB::compute_obb_iou(
    const types::BoxfWithAngle &box1,
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

void TRTYoloV8OBB::nms_obb(
    std::vector<types::BoxfWithAngle> &input,
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

#if LITETRT_DEBUG
    std::cout << "NMS input: " << input.size() << ", output: " << output.size() << "\n";
#endif
}

void TRTYoloV8OBB::detect(
    const cv::Mat &mat,
    std::vector<types::BoxfWithAngle> &detected_boxes,
    float score_threshold,
    float iou_threshold,
    unsigned int topk)
{

    if (mat.empty())
        return;

    int img_height = static_cast<int>(mat.rows);
    int img_width = static_cast<int>(mat.cols);

    // Preprocessing
    cv::Mat mat_rs = mat.clone();
    preprocess(mat_rs);

    // 1. Prepare input tensor
    std::vector<float> input;
    trtcv::utils::transform::create_tensor(mat_rs, input, input_node_dims, trtcv::utils::transform::CHW);

    // 2. TensorRT inference
    cudaMemcpyAsync(buffers[0], input.data(),
                    input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    bool status = trt_context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    if (!status)
    {
        std::cerr << "Failed to infer by TensorRT." << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);

    // 3. Get output
    auto pred_dims = output_node_dims[0];
    std::vector<float> output(pred_dims[0] * pred_dims[1] * pred_dims[2]);

    cudaMemcpyAsync(output.data(), buffers[1],
                    pred_dims[0] * pred_dims[1] * pred_dims[2] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 4. Post-processing
    std::vector<types::BoxfWithAngle> bbox_collection;
    generate_bboxes_obb(bbox_collection, output.data(), score_threshold, img_height, img_width);

    // 5. NMS
    nms_obb(bbox_collection, detected_boxes, iou_threshold, topk);
}
