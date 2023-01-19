#include "yolov8_radar.h"

#define VIDEOS_PATH "/home/knight/Sharefolder_Knight/test_R2_small.mp4"
std::vector<yolo_radar_trt::Object_result> result;

int main() {
    shared_ptr <yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V8); //预处理
    cv::Mat frame;
    cv::VideoCapture cap(VIDEOS_PATH);
    if (!cap.isOpened()) {
        printf("Could no find Camera.\n");
        return 0;
    }
    while (true) {
        cap.read(frame);
        if (frame.empty())
            break;
        result = yolo_radar_trt::work(yolo, frame);
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.bbox << std::endl;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
