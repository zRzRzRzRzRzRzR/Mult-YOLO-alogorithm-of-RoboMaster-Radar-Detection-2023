#ifndef OPENVINO_RADAR_CLASS_YOLOV7_RADAR_H
#define OPENVINO_RADAR_CLASS_YOLOV7_RADAR_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include <inference_engine.hpp>

#define NMS_THRESHOLD 0.20f //NMS参数
#define CONF_THRESHOLD 0.70f //置信度参数
#define CONF_REMAIN 0.05 //保留一帧保留的权重比例
#define IMG_SIZE 640  //推理图像大小，如果不是640 和 416 需要自己在下面添加anchor
#define ANCHOR_SMALL 0 //小目标层数
#define ANCHOR_BIG 0 //大目标层数
#define DEBUG //是否开启debug模式，开启以后输出推理过程和帧数
#define VIDEOS //是否展示推理视频
#define SOFT_NMS //是否采用soft_nms方法
#define CLS_NUM 14 // 种类数量，请在下面的class_names 补充清楚对应的数量
#define DEVICE "CPU" // 设备选择
#define MODEL_PATH "/home/zr/C++_inference_openvino_radar/demo_inference/gmaster_demo_640_yolov7_radar/gmaster_demo_640_yolov7_radar.xml"

class yolo_radar {
public:
    yolo_radar();

    struct Object {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };
    struct Object_result {
        cv::Rect_<float> bbox;
        int label;
        float prob;
    };

    static cv::Mat letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd);

    cv::Rect scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h);

    void drawPred(int classId, float conf, cv::Rect box, cv::Mat &frame, const std::vector <std::string> &classes);

    static void generate_proposals(int stride, const float *feat, std::vector <Object> &objects);

    std::vector <Object_result> work(cv::Mat src_img);

private:

#ifdef DEBUG
    cv::TickMeter meter;
#endif
    ov::Core core;

    std::shared_ptr <ov::Model> model;

    ov::CompiledModel compiled_model;

    ov::InferRequest infer_request;

    ov::Tensor input_tensor1;

    const std::vector <std::string> class_names = {
            "B1", "B2", "B3", "B4", "B5", "BO", "BS", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };

    static float sigmoid(float x) {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

};

#endif //OPENVINO_RADAR_CLASS_YOLOV7_RADAR_H
