//
// Created by wangzijian on 24-7-11.
//

#include "trt_handler.h"

using trtcore::BasicTRTHandler;  // using namespace

BasicTRTHandler::BasicTRTHandler(const std::string &_trt_model_path,
                                 unsigned int _num_threads)
    : log_id(_trt_model_path.data()), num_threads(_num_threads)
{
    trt_model_path = _trt_model_path.data();  // model path
    initialize_handler();
    print_debug_string();
}

BasicTRTHandler::~BasicTRTHandler()
{
    // don't need free by manunly
    for (auto buffer : buffers)
    {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);
}

void BasicTRTHandler::initialize_handler()
{
    // read engine file
    std::ifstream file(trt_model_path, std::ios::binary);

    if (!file.good())
    {
        std::cerr << "Failed to read model file: " << trt_model_path
                  << std::endl;
        return;
    }
    file.seekg(0, std::ifstream::end);
    size_t total_file_size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    // Check for Ultralytics metadata header (4-byte length + JSON metadata)
    // Format: [4 bytes: metadata length (little-endian)] [N bytes: JSON] [rest:
    // TensorRT engine]
    int32_t metadata_length = 0;
    file.read(reinterpret_cast<char *>(&metadata_length), sizeof(int32_t));

    size_t model_offset = 0;
    size_t model_size = total_file_size;

    // If metadata_length is reasonable (> 0 and < 100KB), skip the metadata
    if (metadata_length > 0 && metadata_length < 102400)
    {
        // This looks like Ultralytics format with metadata
        model_offset = sizeof(int32_t) + metadata_length;
        model_size = total_file_size - model_offset;
        file.seekg(model_offset, std::ifstream::beg);
#if LITETRT_DEBUG
        std::cout << "Detected Ultralytics metadata header (" << metadata_length
                  << " bytes), skipping to TensorRT engine at offset "
                  << model_offset << std::endl;
#endif
    }
    else
    {
        // No metadata or invalid format, read from beginning
        file.seekg(0, std::ifstream::beg);
    }

    std::vector<char> model_data(model_size);
    file.read(model_data.data(), model_size);
    file.close();

    trt_runtime.reset(nvinfer1::createInferRuntime(trt_logger));
    // engine deserialize
    trt_engine.reset(
        trt_runtime->deserializeCudaEngine(model_data.data(), model_size));
    if (!trt_engine)
    {
        std::cerr << "Failed to deserialize the TensorRT engine." << std::endl;
        return;
    }
    trt_context.reset(trt_engine->createExecutionContext());
    if (!trt_context)
    {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }
    cudaStreamCreate(&stream);

    // make the flexible one input and multi output
    int num_io_tensors =
        trt_engine->getNbIOTensors();  // get the input and output's num
    buffers.resize(num_io_tensors);

#if LITETRT_DEBUG
    std::cout << "Total I/O tensors: " << num_io_tensors << std::endl;
#endif

    for (int i = 0; i < num_io_tensors; ++i)
    {
        auto tensor_name = trt_engine->getIOTensorName(i);
        nvinfer1::Dims tensor_dims = trt_engine->getTensorShape(tensor_name);
        auto io_mode = trt_engine->getTensorIOMode(tensor_name);

#if LITETRT_DEBUG
        std::cout << "Tensor[" << i << "] name=" << tensor_name << " mode="
                  << (io_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT"
                                                                : "OUTPUT")
                  << " shape=[";
        for (int j = 0; j < tensor_dims.nbDims; ++j)
        {
            std::cout << tensor_dims.d[j];
            if (j < tensor_dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
#endif

        // input
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
        {
            size_t tensor_size = 1;
            for (int j = 0; j < tensor_dims.nbDims; ++j)
            {
                tensor_size *= tensor_dims.d[j];
                input_node_dims.push_back(tensor_dims.d[j]);
            }
            cudaMalloc(&buffers[i], tensor_size * sizeof(float));
            trt_context->setTensorAddress(tensor_name, buffers[i]);
            continue;
        }

        // output
        size_t tensor_size = 1;

        std::vector<int64_t> output_node;
        for (int j = 0; j < tensor_dims.nbDims; ++j)
        {
            output_node.push_back(tensor_dims.d[j]);
            tensor_size *= tensor_dims.d[j];
        }
        output_node_dims.push_back(output_node);

        cudaMalloc(&buffers[i], tensor_size * sizeof(float));
        trt_context->setTensorAddress(tensor_name, buffers[i]);
        output_tensor_size++;
    }
}

void BasicTRTHandler::print_debug_string()
{
    std::cout << "TensorRT model loaded from: " << trt_model_path << std::endl;
    std::cout << "Input tensor size: " << input_tensor_size << std::endl;
    std::cout << "Output tensor size: " << output_tensor_size << std::endl;
}
