#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cam(0);
    if (!cam.isOpened()) {
        std::cerr << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    // 해상도 축소
    cam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // 고정된 HSV 범위 (밝은 환경 기준 손색)
    int minH = 0, minS = 20, minV = 100;
    int maxH = 30, maxS = 150, maxV = 255;

    cv::Mat frame, hsv, mask;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> hull;
    std::vector<int> hullIdx;
    std::vector<cv::Vec4i> defects;

    while (true) {
        cam >> frame;
        if (frame.empty()) break;

        // HSV 변환 + 마스크
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), mask);
        cv::medianBlur(mask, mask, 5);

        // 가장 큰 윤곽선 탐색
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int maxIdx = -1;
        double maxArea = 1000.0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = static_cast<int>(i);
            }
        }

        if (maxIdx != -1) {
            cv::convexHull(contours[maxIdx], hull);
            cv::convexHull(contours[maxIdx], hullIdx);
            cv::convexityDefects(contours[maxIdx], hullIdx, defects);

            // 손 중심점 + 박스
            cv::Rect box = cv::boundingRect(hull);
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1);
            cv::Point center(box.x + box.width / 2, box.y + box.height / 2);
            cv::circle(frame, center, 4, cv::Scalar(0, 255, 0), -1);

            // defect 연결선 (간단 표시)
            for (const auto& d : defects) {
                cv::Point p1 = contours[maxIdx][d[0]];
                cv::Point p2 = contours[maxIdx][d[1]];
                cv::Point p3 = contours[maxIdx][d[2]];
                cv::line(frame, p1, p3, cv::Scalar(255, 255, 0), 1);
                cv::line(frame, p3, p2, cv::Scalar(255, 255, 0), 1);
            }
        }

        // 디스플레이
        cv::imshow("Hand Detection", frame);
        if (cv::waitKey(10) == 27) break;  // ESC 키 종료
    }

    return 0;
}