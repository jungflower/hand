// 필요한 헤더들을 포함합니다. 현대적인 C++ API에서는 opencv.hpp 하나로 충분할 때가 많습니다.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

// cv와 std 네임스페이스를 사용합니다.
using namespace cv;
using namespace std;

// GStreamer 파이프라인 문자열 정의
// 파이카메라(libcamerasrc)의 영상을 OpenCV가 처리할 수 있는 형태로 변환합니다.
// 해상도(640x480)나 프레임(30/1)은 필요에 따라 조절할 수 있습니다.
const string gstreamer_pipeline = "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink";

int main()
{
    // 1. 카메라 열기 (GStreamer 파이프라인 사용)
    cv::VideoCapture cap(0, cv::CAP_V4L2); // /dev/video0 카메라 장치 열기
    if(!cap.isOpened()){
        std::cerr << "Camera open failed !!" << std::endl;
        return -1;
    }

    // 관심 영역(ROI) 정의
    Rect roi_rect(340, 100, 270, 270);

    while (true)
    {
        // cap >> frame;
        // if (frame.empty()) break;

        // 2. 프레임 읽기
        Mat frame; // IplImage* 대신 cv::Mat 사용
        cap.read(frame);

        if (frame.empty())
        {
            cout << "프레임이 비어있습니다." << endl;
            break;
        }

        // 3. 관심 영역(ROI) 설정
        // ROI 영역이 프레임 밖으로 나가지 않도록 경계 검사
        if (roi_rect.x + roi_rect.width > frame.cols || roi_rect.y + roi_rect.height > frame.rows) {
            cout << "ROI가 프레임 크기를 벗어납니다. ROI 좌표를 조절하세요." << endl;
            // 화면에 현재 해상도 출력
            putText(frame, "Check ROI Coordinates!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2);
            imshow("Gesture Recognition", frame);
            if (waitKey(10) == 27) break;
            continue;
        }
        Mat roi = frame(roi_rect);

        // 4. 이미지 전처리
        Mat gray, blurred, thresholded;
        cvtColor(roi, gray, COLOR_BGR2GRAY);
        blur(gray, blurred, Size(12, 12));
        threshold(blurred, thresholded, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

        // 5. 윤곽선 찾기
        vector<vector<Point>> contours;
        findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty())
        {
            // 가장 큰 윤곽선 찾기
            double max_area = 0;
            int max_contour_idx = -1;
            for (size_t i = 0; i < contours.size(); i++)
            {
                double area = contourArea(contours[i]);
                if (area > max_area)
                {
                    max_area = area;
                    max_contour_idx = i;
                }
            }

            // 가장 큰 윤곽선이 최소 크기(1000)를 넘을 경우 분석
            if (max_area > 1000)
            {
                vector<int> hull_indices;
                vector<Vec4i> defects;

                // 6. 볼록 껍질(Convex Hull)과 오목 결함(Convexity Defects) 계산
                convexHull(contours[max_contour_idx], hull_indices, false);
                convexityDefects(contours[max_contour_idx], hull_indices, defects);

                int defect_count = 0;
                for (const Vec4i& v : defects)
                {
                    float depth = v[3] / 256.0; // 결함 깊이
                    if (depth > 40) // 깊이가 40 이상인 결함만 카운트
                    {
                        defect_count++;
                        Point far_pt = contours[max_contour_idx][v[2]];
                        circle(roi, far_pt, 5, Scalar(0, 0, 255), -1); // 오목한 지점에 빨간 원 그리기
                    }
                }

                // 7. 텍스트 결정 및 출력
                string text;
                if (defect_count == 1) text = "Hi, This is 2";
                else if (defect_count == 2) text = "This is 3";
                else if (defect_count == 3) text = "Fantastic 4";
                else if (defect_count == 4) text = "It's 5";
                else text = "Jarvis is busy :P";
                
                putText(frame, text, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
            }
        }
        
        // 전체 프레임에 ROI 영역 표시
        rectangle(frame, roi_rect, Scalar(0, 255, 255), 2);
        
        // 8. 결과 보여주기
        imshow("Gesture Recognition", frame);
        // 디버깅용 흑백 이미지 창
        imshow("Thresholded", thresholded);

        // 'ESC' 키를 누르면 종료
        if (waitKey(10) == 27)
        {
            break;
        }
    }

    // 자원 해제
    cap.release();
    destroyAllWindows();
    return 0;
}