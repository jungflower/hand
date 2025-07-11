#include "ModelLoader.hpp"

#include <iostream>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

#define INPUT_NORM_MEAN 127.5f
#define INPUT_NORM_STD  127.5f


hand::ModelLoader::ModelLoader(std::string modelPath) {
    loadModel(modelPath.c_str());
    buildInterpreter();
    allocateTensors();
    fillInputTensors();
    fillOutputTensors();

    m_inputLoads.resize(getNumberOfInputs(), false);
}


std::vector<int> hand::ModelLoader::getInputShape(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].dims;

    return std::vector<int>();
}


float* hand::ModelLoader::getInputData(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].data;

    return nullptr;
}


size_t hand::ModelLoader::getInputSize(int index) const {
    if (isIndexValid(index, 'i'))
        return m_inputs[index].bytes;

    return 0;
}


int hand::ModelLoader::getNumberOfInputs() const {
    return m_inputs.size();
}


std::vector<int> hand::ModelLoader::getOutputShape(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].dims;
        
    return std::vector<int>();
}


float* hand::ModelLoader::getOutputData(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].data;

    return nullptr;
}


size_t hand::ModelLoader::getOutputSize(int index) const {
    if (isIndexValid(index, 'o'))
        return m_outputs[index].bytes;

    return 0;
}


int hand::ModelLoader::getNumberOfOutputs() const {
    return m_outputs.size();
}


void hand::ModelLoader::loadImageToInput(const cv::Mat& inputImage, int idx) {
    if (isIndexValid(idx, 'i')) {
        cv::Mat resizedImage = preprocessImage(inputImage, idx); // Need optimize
        loadBytesToInput(resizedImage.data, idx);
    }
}


void hand::ModelLoader::loadBytesToInput(const void* data, int idx) {
    if (isIndexValid(idx, 'i')) {
        memcpy(m_inputs[idx].data, data, m_inputs[idx].bytes);
        m_inputLoads[idx] = true;
    }
}


void hand::ModelLoader::runInference() {
    inputChecker();
    m_interpreter->Invoke(); // Tflite inference
}


std::vector<float> hand::ModelLoader::loadOutput(int index) const {
    if (isIndexValid(index, 'o')) {
        int n = m_outputs[index].bytes;
        std::vector<float> inference(n);
        memcpy(&(inference[0]), m_outputs[index].data, n);
        return inference;
    }
    return std::vector<float>();
}


//-------------------Private methods start here-------------------

void hand::ModelLoader::loadModel(const char* modelPath) {
    m_model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (m_model == nullptr) {
        std::cerr << "Fail to build FlatBufferModel from file: " << modelPath << std::endl;
        std::exit(1);
    }  
}


void hand::ModelLoader::buildInterpreter(int numThreads) {
    tflite::ops::builtin::BuiltinOpResolver resolver;

    if (tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter." << std::endl;
        std::exit(1);
    }
    m_interpreter->SetNumThreads(numThreads);
}


void hand::ModelLoader::allocateTensors() {
    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors." << std::endl;
        std::exit(1);
    }
}


void hand::ModelLoader::fillInputTensors() {
    for (auto input: m_interpreter->inputs()) {
        TfLiteTensor* inputTensor =  m_interpreter->tensor(input);
        TfLiteIntArray* dims =  inputTensor->dims;

        m_inputs.push_back({
            inputTensor->data.f,
            inputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}


void hand::ModelLoader::fillOutputTensors() {
    for (auto output: m_interpreter->outputs()) {
        TfLiteTensor* outputTensor =  m_interpreter->tensor(output);
        TfLiteIntArray* dims =  outputTensor->dims;

        m_outputs.push_back({
            outputTensor->data.f,
            outputTensor->bytes,
            dims->data,
            dims->size
        });
    }
}


bool hand::ModelLoader::isIndexValid(int idx, const char c) const {
    int size = 0;
    if (c == 'i')
        size = m_inputs.size();
    else if (c == 'o')
        size = m_outputs.size();
    else 
        return false;

    if (idx < 0 || idx >= size) {
        std::cerr << "Index " << idx << " is out of range (" \
        << size << ")." << std::endl;
        return false;
    }
    return true;
}


bool hand::ModelLoader::isAllInputsLoaded() const {
    return (
        std::find(m_inputLoads.begin(), m_inputLoads.end(), false)
     == m_inputLoads.end()); 
}


void hand::ModelLoader::inputChecker() {
    if (isAllInputsLoaded() == false) {
        std::cerr << "Input ";
        for (int i = 0; i < m_inputLoads.size(); ++i) {
            if (m_inputLoads[i] == false) {
                std::cerr << i << " ";
            }
        }
        std::cerr << "haven't been loaded." << std::endl;
        std::exit(1);
    }
    std::fill(m_inputLoads.begin(), m_inputLoads.end(), false);
}


cv::Mat hand::ModelLoader::preprocessImage(const cv::Mat& in, int idx) const {
    auto out = convertToRGB(in);

    std::vector<int> inputShape = getInputShape(idx);
    int H = inputShape[1];
    int W = inputShape[2]; 

    cv::Size wantedSize = cv::Size(W, H);
    cv::resize(out, out, wantedSize);

    /*
    Equivalent to (out - mean)/ std
    */
    out.convertTo(out, CV_32FC3, 1 / INPUT_NORM_STD, -INPUT_NORM_MEAN / INPUT_NORM_STD);
    return out;
}


cv::Mat hand::ModelLoader::convertToRGB(const cv::Mat& in) const {
    cv::Mat out;
    int type = in.type();

    if (type == CV_8UC3) {
        cv::cvtColor(in, out, cv::COLOR_BGR2RGB);
    }
    else if (type == CV_8UC4) {
        cv::cvtColor(in, out, cv::COLOR_BGRA2RGB);
    }
    else {
        std::cerr << "Image of type " << type << " not supported" << std::endl;
        std::exit(1);
    }
    return out;
}