#include <opencv2/opencv.hpp>
#include <iostream>

int main() {

    //cv::VideoCapture cap(0, cv::CAP_V4L2);
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "카메라 열기 실패" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, avgFloat, avgGray, diff, thresh;
    bool isAvgInit = false;
    const int MOTION_THRESHOLD = 6;
    const int MOTION_TRIGGER_SCORE = 4;
    const int DISPLAY_TEXT_FRAMES = 3000;

    int motionScore = 0;
    std::string direction = "";
    int displayCounter = 0;
    int prevMinX = -1;
    int prevMaxX = -1;

    // STOP 인식 -----------
    bool palmDetected = false;
    int lastMotionArea = 0;
    int palmStableCounter = 0;

    const float PALM_AREA_RATIO_THRESHOLD = 2.5f;
    const int PALM_MIN_AREA = 10000;
    const int PALM_RESET_FRAME_COUNT = 30;
    // -------------------

    // threshold 조절
    int threshold_value = 40;
    cv::namedWindow("Thresh");
    cv::createTrackbar("Threshold", "Thresh", &threshold_value, 100); // 0 ~ 100 조절 가능
    // -----------------------

    while (true) {
        bool justDetected = false;
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
        cv::threshold(diff, thresh, threshold_value, 255, cv::THRESH_BINARY);
        cv::erode(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 2);

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
            //std::cout << "min: " << minX << ", max: " << maxX << std::endl;

            // 손바닥 판정
            int area = motionPoints.size();
            // 손바닥 등장 판단: area가 갑자기 커졌을 때, 일정 이상일 때
            if (!palmDetected && lastMotionArea > 0) {
                float areaRatio = (float)area / lastMotionArea;
                if (areaRatio > PALM_AREA_RATIO_THRESHOLD && area > PALM_MIN_AREA) {
                    std::cout << "STOP" << std::endl;
                    palmDetected = true;
                    palmStableCounter = 0;
                }
            }

            // 손바닥 유지 시간 체크
            if (palmDetected) {
                palmStableCounter++;
                if (palmStableCounter > PALM_RESET_FRAME_COUNT) {
                    palmDetected = false;
                    palmStableCounter = 0;
                }
            }
            /*
            if (!palmDetected && lastMotionArea < 5000 && area > 12000) {
                std::cout << "STOP" << std::endl;
                palmDetected = true;
            }

            if (palmDetected && area < 3000) {
                palmDetected = false;
            }
            */
            lastMotionArea = area;

            // left / right swipe 판정 (손바닥 상태 아닐때만)
            if (!palmDetected) {
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
                        //motionScore = 0;
                        if (motionScore > 0) motionScore--;
                        else if (motionScore < 0) motionScore++;
                    }

                    if (motionScore >= MOTION_TRIGGER_SCORE) {
                        std::cout << "LEFT" << std::endl;
                        motionScore = 0;
                        prevMinX = -1;
                        prevMaxX = -1;
                        justDetected = true;
                    }
                    else if (motionScore <= -MOTION_TRIGGER_SCORE) {
                        std::cout << "RIGHT" << std::endl;
                        motionScore = 0;
                        prevMinX = -1;
                        prevMaxX = -1;
                        justDetected = true;
                    }
                }
                if (!justDetected) {
                    prevMinX = minX;
                    prevMaxX = maxX;
                }
            }
            else {
                prevMinX = minX;
                prevMaxX = maxX;
            }
        }
        else {
            // 손 없을 때만 점수 초기화
            if (!palmDetected)   motionScore = 0;
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