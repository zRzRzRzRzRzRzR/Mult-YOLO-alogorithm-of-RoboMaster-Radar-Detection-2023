#include "yolov7_radar.h"

#define VIDEO_PATH "/home/zr/C++_inference_openvino_radar/demo_inference/test_for_radar.mp4"
yolo_radar DEMO;
std::vector <yolo_radar::Object_result> result;

int main() {
    cv::VideoCapture cap;
    cap.open(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cout << "相机没有打开" << std::endl;
        return 0;
    }
    while (true) {
        cv::Mat src_img;
        bool ret = cap.read(src_img);
        if (!ret) break;
        result = DEMO.work(src_img);
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.bbox << std::endl;
        }
    }
}