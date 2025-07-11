#include "Handlandmark.hpp"
#include <iostream>

#define HAND_LANDMARKS 21
/*
Helper function
*/
bool __isIndexValid(int idx) {
    if (idx < 0 || idx >= HAND_LANDMARKS) {
        std::cerr << "Index " << idx << " is out of range (" \
        << HAND_LANDMARKS << ")." << std::endl;
        return false;
    }
    return true;
}


hand::HandLandmark::HandLandmark(std::string modelPath) :
    HandDetection(modelPath),
    m_landmarkModel(modelPath + std::string("/hand_landmark_full.tflite")) // Just change the model path
{}


void hand::HandLandmark::runInference() {
    HandDetection::runInference();
    auto roi = HandDetection::getHandRoi();
    if (roi.empty()) return;

    auto Hand = HandDetection::cropFrame(roi);
    m_landmarkModel.loadImageToInput(Hand);
    m_landmarkModel.runInference();
}


cv::Point hand::HandLandmark::getHandLandmarkAt(int index) const {
    if (__isIndexValid(index)) {
        auto roi = HandDetection::getHandRoi();

        float _x = m_landmarkModel.getOutputData()[index * 3];
        float _y = m_landmarkModel.getOutputData()[index * 3 + 1];
        //float _z = m_landmarkModel.getOutputData()[index * 3 + 2];

        int x = (int)(_x / m_landmarkModel.getInputShape()[2] * roi.width) + roi.x;
        int y = (int)(_y / m_landmarkModel.getInputShape()[1] * roi.height) + roi.y;

        //std::cout << "z: " << _z << std::endl;

        return cv::Point(x,y);
    }
    return cv::Point();
}


std::vector<cv::Point> hand::HandLandmark::getAllHandLandmarks() const {
    if (HandDetection::getHandRoi().empty())
        return std::vector<cv::Point>();

    std::vector<cv::Point> landmarks(HAND_LANDMARKS);
    for (int i = 0; i < HAND_LANDMARKS; ++i) {
        landmarks[i] = getHandLandmarkAt(i);
    }
    return landmarks;
}


std::vector<float> hand::HandLandmark::loadOutput(int index) const {
    return m_landmarkModel.loadOutput();
}

