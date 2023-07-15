#include "include/yolov8_radar.h"
#include <string>
#include <condition_variable>
#include <queue>
#include <dirent.h>

void yolo_radar_trt::drawPred(int classId, float conf, cv::Rect box, cv::Mat &frame,
                              const std::vector<std::string> &classes) {
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 1);
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int) classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                  cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
}

shared_ptr<yolo_radar_trt::Infer> yolo_radar_trt::prepare(yolo_radar_trt::Type type) {
    yolo_radar_trt::set_device(GPU_ID);
    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::create_infer(MODEL_PATH, type, GPU_ID, CONF_THRESHOLD, NMS_THRESHOLD);
    if (yolo == nullptr) {
        printf("Could no find YoloV8 model.\n");
        return NULL;
    }
    return yolo;
}

vector<yolo_radar_trt::Object> yolo_radar_trt::work(shared_ptr<yolo_radar_trt::Infer> yolo, Mat frame,string frame_name) {
    vector<yolo_radar_trt::Object> object_result;
#ifdef DEBUG
    cv::TickMeter meter;
    meter.start();
#endif
    auto boxes = yolo->commit(frame).get();
    for (auto &obj: boxes) {
#ifdef VIDEOS
        drawPred(obj.label, obj.prob, obj.rect, frame, yolo_radar_trt::class_names);
#endif
        obj.depth = 0;
        object_result.push_back(obj);
    }

#ifdef VIDEOS
    cv::namedWindow(frame_name, cv::WINDOW_NORMAL);  // 创建一个可以调整大小的窗口
    cv::resizeWindow(frame_name, 640, 512);  // 设定窗口的大小
    imshow(frame_name, frame);
    cv::waitKey(1);
#endif
#ifdef DEBUG
    meter.stop();
//    printf("Time: %f\n", meter.getTimeMilli());
    meter.reset();
#endif
    return object_result;
}


