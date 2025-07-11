#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, avgFloat, avgGray, diff, thresh;
    bool isAvgInit = false;
    const int MOTION_THRESHOLD = 10;
    const int MOTION_TRIGGER_SCORE = 5;
    const int DISPLAY_TEXT_FRAMES = 3000;

    int motionScore = 0;
    std::string direction = "";
    int displayCounter = 0;
    int prevMinX = -1;
    int prevMaxX = -1;

    bool palmDetected = false;
    int lastMotionArea = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 1. 전처리
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // 2. 평균 초기화
        if (!isAvgInit) {
            gray.convertTo(avgFloat, CV_32F);
            isAvgInit = true;
        }

        // 3. 평균 업데이트
        cv::accumulateWeighted(gray, avgFloat, 0.05);
        avgFloat.convertTo(avgGray, CV_8U);

        // 4. 변화 감지
        cv::absdiff(gray, avgGray, diff);
        cv::threshold(diff, thresh, 40, 255, cv::THRESH_BINARY);
        cv::erode(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2); // 침식
        cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2); // 팽창

        // 5. minX, maxX 찾기
        std::vector<cv::Point> motionPoints;
        cv::findNonZero(thresh, motionPoints);
        
        if (!motionPoints.empty()) {
            int minX = frame.cols;
            int maxX = 0;
            // 가장 왼쪽 오른쪽 좌표 구함
            for (const auto& pt : motionPoints) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
            }
            // std::cout << "min: " << minX << ", max: " << maxX << std::endl;

            int area = motionPoints.size();
            if(!palmDetected && lastMotionArea < 5000 && area > 12000) {
                std::cout << "STOP" << std::endl;
                palmDetected = true;
            }

            if(palmDetected && area < 3000){
                palmDetected = false;
            }

            lastMotionArea = area;
            
            // left / right swipe 판정
            if (prevMinX != -1 && prevMaxX != -1) {
                int dxMax = maxX - prevMaxX;
                int dxMin = minX - prevMinX;
                // std::cout << dxMin << ", " << dxMax << std::endl;
                if (dxMax > MOTION_THRESHOLD) {
                    motionScore = (motionScore >= 0) ? motionScore + 1 : 0;
                }
                else if(dxMin < -MOTION_THRESHOLD) {
                  motionScore = (motionScore <= 0) ? motionScore - 1 : 0;
                }
                else {
                    motionScore = 0;
                }

                if (motionScore >= MOTION_TRIGGER_SCORE) {
                  std::cout << "LEFT" << std::endl;
                    motionScore = 0;
                }
                else if(motionScore <= -MOTION_TRIGGER_SCORE) {
                  std::cout << "RIGHT" << std::endl;
                  motionScore = 0;
                }
            }
            prevMinX = minX;
            prevMaxX = maxX;
        } else {
            motionScore = 0;
        }

        // 7. 영상 출력
        cv::imshow("Camera", frame);

        // cv::imshow("Diff", diff);
        cv::imshow("Thresh", thresh);
        /*
        cv::Mat thresh_color, combined;
        cv::cvtColor(thresh, thresh_color, cv::COLOR_GRAY2BGR);  // thresh(1채널) → BGR(3채널)
        if (thresh_color.size() != frame.size()) {
            cv::resize(thresh_color, thresh_color, frame.size());
        }
        cv::hconcat(thresh_color, frame, combined);
        cv::imshow("Hand Detection (Mask | Original)", combined);
        */
        if (cv::waitKey(30) == 27) break;  // ESC
    }
    return 0;
}