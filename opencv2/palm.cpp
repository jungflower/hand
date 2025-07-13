#include <opencv2/opencv.hpp>
#include <iostream>

enum GestureState {
    IDLE,
    STOP_DETECTED,
    SWIPE_TRACKING
};

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, avgFloat, avgGray, diff, thresh;
    bool isAvgInit = false;
    const int MOTION_THRESHOLD = 6;
    const int MOTION_TRIGGER_SCORE = 4;
    const int PALM_RESET_FRAME_COUNT = 30;
    const float PALM_AREA_RATIO_THRESHOLD = 2.5f;
    const int PALM_MIN_AREA = 10000;

    int motionScore = 0;
    int prevMinX = -1, prevMaxX = -1;
    int lastMotionArea = 0;
    int palmStableCounter = 0;
    GestureState state = IDLE;

    int threshold_value = 40;
    cv::namedWindow("Thresh");
    cv::createTrackbar("Threshold", "Thresh", &threshold_value, 100);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        if (!isAvgInit) {
            gray.convertTo(avgFloat, CV_32F);
            isAvgInit = true;
        }

        cv::accumulateWeighted(gray, avgFloat, 0.05);
        avgFloat.convertTo(avgGray, CV_8U);

        cv::absdiff(gray, avgGray, diff);
        cv::threshold(diff, thresh, threshold_value, 255, cv::THRESH_BINARY);
        cv::erode(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<cv::Point> motionPoints;
        cv::findNonZero(thresh, motionPoints);

        if (!motionPoints.empty()) {
            int minX = frame.cols;
            int maxX = 0;
            for (const auto& pt : motionPoints) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
            }

            int area = motionPoints.size();
            float areaRatio = lastMotionArea > 0 ? (float)area / lastMotionArea : 0;

            switch (state) {
            case IDLE:
                if (areaRatio > PALM_AREA_RATIO_THRESHOLD && area > PALM_MIN_AREA) {
                    std::cout << "STOP" << std::endl;
                    state = STOP_DETECTED;
                    palmStableCounter = 0;
                }
                else {
                    state = SWIPE_TRACKING;
                    motionScore = 0;
                }
                break;

            case STOP_DETECTED:
                palmStableCounter++;
                if (palmStableCounter > PALM_RESET_FRAME_COUNT) {
                    state = IDLE;
                    palmStableCounter = 0;
                }
                break;

            case SWIPE_TRACKING:
                if (prevMinX != -1 && prevMaxX != -1) {
                    int dxMax = maxX - prevMaxX;
                    int dxMin = minX - prevMinX;
                    std::cout << dxMin << ", " << dxMax << std::endl;

                    if (dxMax > MOTION_THRESHOLD) {
                        motionScore = (motionScore >= 0) ? motionScore + 1 : 0;
                    }
                    else if (dxMin < -MOTION_THRESHOLD) {
                        motionScore = (motionScore <= 0) ? motionScore - 1 : 0;
                    }
                    else {
                        if (motionScore > 0) motionScore--;
                        else if (motionScore < 0) motionScore++;
                    }

                    if (motionScore >= MOTION_TRIGGER_SCORE) {
                        std::cout << "LEFT" << std::endl;
                        motionScore = 0;
                        state = IDLE;
                    }
                    else if (motionScore <= -MOTION_TRIGGER_SCORE) {
                        std::cout << "RIGHT" << std::endl;
                        motionScore = 0;
                        state = IDLE;
                    }
                }
                prevMinX = minX;
                prevMaxX = maxX;
                break;
            }
            lastMotionArea = area;
        }
        else {
            motionScore = 0;
            if (state == SWIPE_TRACKING)
                state = IDLE;
        }

        cv::imshow("Camera", frame);
        cv::imshow("Thresh", thresh);
        if (cv::waitKey(30) == 27) break;
    }
    return 0;
}
