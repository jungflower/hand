#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

cv::Rect roiBox(200, 100, 300, 300); // 초기 ROI
cv::Point prevCenter(-1, -1);       // 이전 중심점
int swipeThreshold = 50;            // 스와이프 감지 민감도

// 얼굴 영역 제거
void removeFaceArea(cv::Mat& img, cv::CascadeClassifier& cascade) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(gray, faces, 1.2, 5, 0, cv::Size(30, 30));

    int height = img.rows;
    for (const cv::Rect& r : faces) {
        int cx = r.x + r.width / 2;
        int cy = r.y + r.height / 2;
        int w = r.width * 1.0;
        int h = r.height * 1.0;
        cv::rectangle(img,
                      cv::Point(cx - w / 2, cy - h / 2),
                      cv::Point(cx + w / 2, cy + h / 2),
                      cv::Scalar(0, 0, 0), cv::FILLED);
    }
}

// 손 피부색 기반으로 마스크 생성
cv::Mat makeHandMask(const cv::Mat& img_bgr) {
    cv::Mat img_hsv;
    cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower(0, 20, 70);
    cv::Scalar upper(20, 180, 255);
    cv::Mat mask;
    cv::inRange(img_hsv, lower, upper, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    return mask;
}

int findLargestContour(const std::vector<std::vector<cv::Point>>& contours) {
    int maxIdx = -1;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > 1000 && area < 100000 && area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}

// Swipe 함수
void detectSwipe(const cv::Point& center, const cv::Point& prevCenter, int threshold, cv::Mat& img) {
    if (prevCenter.x == -1) return;
    int dx = center.x - prevCenter.x;
    if (std::abs(dx) > threshold) {
        std::string direction = (dx > 0) ? "Right Swipe" : "Left Swipe";
        cv::putText(img, direction, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
}

// FPS 표시
void drawFPS(double fps, cv::Mat& img) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << fps;
    cv::putText(img, "FPS: " + ss.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}

// 이미지 합치기
cv::Mat combineImages(const cv::Mat& img, const cv::Mat& mask) {
    cv::Mat mask_color, combined;
    cv::cvtColor(mask, mask_color, cv::COLOR_GRAY2BGR);
    cv::hconcat(img, mask_color, combined);
    return combined;
}

int main() {
    cv::CascadeClassifier face_cascade;
    std::string face_model = "haarcascade_frontalface_alt.xml";

    if (!face_cascade.load(face_model)) {
        std::cerr << "얼굴 검출 모델 로딩 실패" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Camera open failed !!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat img = frame.clone();
        auto start = std::chrono::high_resolution_clock::now();

        removeFaceArea(img, face_cascade);
        cv::Mat mask = makeHandMask(img);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(img, contours, -1, cv::Scalar(0, 255, 255), 2);

        int maxIdx = findLargestContour(contours);
        if (maxIdx != -1) {
            cv::Rect handROI = cv::boundingRect(contours[maxIdx]);
            rectangle(img, handROI, cv::Scalar(0, 255, 0), 2);

            cv::Point center(handROI.x + handROI.width / 2, handROI.y + handROI.height / 2);
            circle(img, center, 5, cv::Scalar(255, 255, 0), cv::FILLED);

            //detectSwipe(center, prevCenter, swipeThreshold, img);
            prevCenter = center;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        drawFPS(fps, img);
        cv::Mat combined = combineImages(img, mask);
        cv::imshow("Hand Detection (Original + Mask)", combined);

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
