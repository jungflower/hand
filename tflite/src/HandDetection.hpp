#ifndef HandDETECTION_H
#define HandDETECTION_H

#include "ModelLoader.hpp"
#include "DetectionPostProcess.hpp"

namespace hand {

    /*
    A model wrapper to use Mediapipe Hand Detector.
    This class is non-copyable.
    */
    class HandDetection : public hand::ModelLoader {
        public:
            /*
            Users MUST provide the FOLDER contain Hand_detection_short.tflite, NOT THE FILE itself.
            */
            HandDetection(std::string modelPath);
            virtual ~HandDetection() = default;

            /*
            Get access to original input image
            */
            cv::Mat getOriginalImage() const;

            /*
            Get the regressor result (first output tensor).
            */
            std::vector<float> getHandRegressor() const;

            /*
            Get the classificator result (second output tensor).
            */         
            std::vector<float> getHandClassificator() const;

            /*
            Get the position of the HIGHEST CONFIDENT Hand
            (Note: the position is relative to the image passed to InputTensor(0))
            */
            virtual cv::Rect getHandRoi() const;

            /*
            Override function from ModelLoader.
            (Note: index does not matter, the model always load to InputTensor(0))
            */
            virtual void loadImageToInput(const cv::Mat& inputImage, int index = 0);       

            /*
            Override function from ModelLoader.
            Can only run when all input tensors have been loaded.
            */
            virtual void runInference();

            /*
            Crop input frame at roi (padding if need)
            */
            cv::Mat cropFrame(const cv::Rect& roi) const;


        private:
            /*
            Override function from ModelLoader.
            This class can only load image to input.
            */
            using ModelLoader::loadBytesToInput;

            /*       
            Convert Detection box back to original size
            */
            cv::Rect calculateRoiFromDetection(const Detection& detection) const;


        private:
            /*
            Help getting Region of Interest from model outputs
            */
            DetectionPostProcess m_postProcessor;

            /*
            Save some informations
            */
            cv::Mat m_originImage;
            cv::Rect m_roi;
    };
}
#endif // HandDETECTION_H