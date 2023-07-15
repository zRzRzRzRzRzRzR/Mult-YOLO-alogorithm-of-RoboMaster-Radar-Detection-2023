#ifndef RADAR_INFER_DEPTH_RADAR_H
#define RADAR_INFER_DEPTH_RADAR_H
#include "yolov8_radar.h"
#define BASELINE 800.0 // ����
#define FOCAL_LENGTH 12.0 //����
#define PIXEL_SIZE 0.004 // ��Ԫ����
#define KNOWN_DEPTH1 18191.0 // ǰ��վ
#define KNOWN_DEPTH2 27565.0 // ����
class DepthRadar {
public:
    cv::Point getCenterPoint(yolo_radar_trt::Object res);
    vector<yolo_radar_trt::Object> processFrames(shared_ptr<yolo_radar_trt::Infer> yolo, cv::Mat left_frame, cv::Mat right_frame);
};
#endif //RADAR_INFER_DEPTH_RADAR_H
