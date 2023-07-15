#include "include/yolov8_radar.h"
#include "include/depth_radar.h"
#define LEFT_VIDEOS_PATH "/media/zr/Data/RoboMaster_data/RMvideos/radar_record_2023/2_video_left.avi"
#define RIGHT_VIDEOS_PATH "/media/zr/Data/RoboMaster_data/RMvideos/radar_record_2023/2_video_right.avi"
std::vector<yolo_radar_trt::Object> result;
void processVideoFrames() {
    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V8); // 预处理
    cv::Mat left_frame, right_frame;
    cv::VideoCapture left_cap(LEFT_VIDEOS_PATH);
    cv::VideoCapture right_cap(RIGHT_VIDEOS_PATH);
    if (!left_cap.isOpened() || !right_cap.isOpened()) {
        printf("Could not find Camera.\n");
        return;
    }

    // 创建DepthRadar类的实例
    DepthRadar radar;

    while (true) {
        left_cap.read(left_frame);
        right_cap.read(right_frame);
        if (left_frame.empty() || right_frame.empty())
            break;
        // 使用DepthRadar类的实例来调用processFrames函数
        result = radar.processFrames(yolo, left_frame, right_frame);
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.rect << std::endl;
            std::cout << "depth:" << i.depth << std::endl;
        }
    }
    left_cap.release();
    right_cap.release();
    cv::destroyAllWindows();
}

int main() {
    processVideoFrames();
    return 0;
}
