#ifndef TENSORRT_RADAR_CLASS_YOLOV8_RADAR_H
#define TENSORRT_RADAR_CLASS_YOLOV8_RADAR_H

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


//caculate deep


#define GPU_ID 0 //显卡编号
#define NMS_THRESHOLD 0.45f //NMS参数
#define CONF_THRESHOLD 0.50f //置信度参数
#define DEBUG //是否开启debug模式，开启以后输出推理过程和帧数
#define VIDEOS //是否展示推理视频
#define MODEL_PATH "../config/radar_0715_grid.trt" //网络位置，格式为 yourmodel.精度.trtmodel
using namespace std;
using namespace cv;

namespace yolo_radar_trt {
    enum class Type : int {
        V7 = 0,
        V8 = 1
    };
    struct Object {
        cv::Rect_<float> rect;
        int label;
        float prob;
        float depth;

        Object() = default;

        Object(cv::Rect_<float> rect, int label, float prob)
                : rect(rect), label(label), prob(prob) {}
    };
    typedef std::vector<Object> ObjectArray;

    class Infer {
    public:
        virtual shared_future<ObjectArray> commit(const cv::Mat &image) = 0;
    };

    shared_ptr<Infer> prepare(Type type);

    vector<Object> work(shared_ptr<Infer> yolo, Mat frame,string frame_name);

    void drawPred(int classId, float conf, cv::Rect box, cv::Mat &frame, const std::vector<std::string> &classes);

    void set_device(int device_id);

    shared_ptr<Infer>
    create_infer(const string &engine_file, Type type, int gpuid, float confidence_threshold, float nms_threshold);

    const std::vector<std::string> class_names = {
            "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS","BB","RB"
    };


};
#endif TENSORRT_RADAR_CLASS_YOLOV8_RADAR_H