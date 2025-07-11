#ifndef HANDLANDMARK_H
#define HANDLANDMARK_H

#include "HandDetection.hpp"

#include <bitset>
#include <vector>

namespace hand {

    class HandLandmark : public hand::HandDetection {
        public:
            /*
            Users MUST provide the FOLDER contain ALL the face_detection_short.tflite, 
            face_landmark.tflite and iris_landmark.tflite 
            */
            HandLandmark(std::string modelPath);
            virtual ~HandLandmark() = default; 

            /*
            Override function from FaceLandmark
            */
            virtual void runInference();

            /*
            Get a landmark from output (index must be in range 0-467)
            The position is relative to the input image at InputTensor(0)
            */
            virtual cv::Point getHandLandmarkAt(int index) const;

            /*
            Get all landmarks from output, which is a vector of length 468 * 3 * 4 (although the first 468 * 3 are enough).
            (Note: index does not matter, it always lgetHandLandmarkAtoad from OutputTensor(0))
            Each landmark is represented by x, y, z(depth), which are raw outputs from Mediapipe Face Landmark model.
            If you want to get relative position to input image, use getAllFaceLandmarks() or getFaceLandmarkAt()
            */
            virtual std::vector<cv::Point> getAllHandLandmarks() const;

            /*
            Get all landmarks from output (index = 0: Eye landmarks, index != 0: Iris landmarks)
            Each landmark is represented by x, y, z(depth), which are raw outputs from Mediapipe Iris Landmark model.
            If you want to get relatives position to input image, use getAllHandLandmarks() or getAllHandLandmark()
            */
            virtual std::vector<float> loadOutput(int index = 0) const;

        private:
            hand::ModelLoader m_landmarkModel;

    };
}
#endif // HANDLANDMARK_H