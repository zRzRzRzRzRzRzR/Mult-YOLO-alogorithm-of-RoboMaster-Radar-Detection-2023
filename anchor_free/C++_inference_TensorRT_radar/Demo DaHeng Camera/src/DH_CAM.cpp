#include "DH_CAM.h"
#include "yolov8_radar.h"
using namespace std;
using namespace cv;
Mat img_src1;
Mat img_src2;
Mat img_src_car;
Camera *camera_left = nullptr;
Camera *camera_right = nullptr;
Camera *camera_car = nullptr;
std::vector<yolo_radar_trt::Object_result> result;
std::vector<yolo_radar_trt::Object_result> result_left;
std::vector<yolo_radar_trt::Object_result> result_right;

inline static std::string getCurrentTime() {
    std::time_t result = std::time(nullptr);
    std::string ret;
    ret.resize(64);
    int wsize = sprintf((char *) &ret[0], "%s", std::ctime(&result));
    ret.resize(wsize);
    return ret;
}

void radar() {
    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V7); //预处理
    string path_left = REORD_PATH + getCurrentTime() + "__left.avi";
    string path_right = REORD_PATH + getCurrentTime() + "__right.avi";
    cv::VideoWriter writer_left(path_left, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 1024));
    cv::VideoWriter writer_right(path_right, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1280, 1024));
    camera_left = new DHCamera(LEFT_SN);
    camera_right = new DHCamera(RIGHT_SN);
    camera_left->init(0, 0, 1280, 1024, 20000, 10, false);
    camera_right->init(0, 0, 1280, 1024, 20000, 10, false);
    while (waitKey(1) != 27) {
        if ((!camera_left->start()) || (!camera_right->start())) {
            cout << "NO Camera or Just one,please check" << endl;
            return;
        }
        camera_left->read(img_src1);
        camera_right->read(img_src2);
        if (img_src1.empty() || img_src2.empty()) {
            cout << "IMG IS EMPTY" << endl;
            return;
        }
        result_left = yolo_radar_trt::work(yolo, img_src1);
        for (auto i: result_left) {
            std::cout << "Find a target" << std::endl;
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox_left:" << i.bbox << std::endl;
        } //左相机

        result_right = yolo_radar_trt::work(yolo, img_src2);
        for (auto i: result_right) {
            std::cout << "bbox_right:" << i.bbox << std::endl;
        } // 右相机

#ifdef VIDEO
        imshow("LEFT", img_src1);
        imshow("RIGHT", img_src2);
        waitKey(1);
#endif
#ifdef RECORD_1
        writer_left << img_src1;
        writer_right << img_src2;
#endif
    }
    return;
}

void car() {
    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V7); //预处理
    VideoCapture cap_car(0);
#if RADAR_MODE == true
    string path_car = REORD_PATH + getCurrentTime() + "__car.avi";
    cv::VideoWriter writer_car(path_car, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 384));
#endif
    camera_car = new DHCamera(CAR_SN);
    camera_car->init(0, 0, 640, 384, 10000, 10, false);
    while (waitKey(1) != 27) {
        if (!camera_car->start()) {
            cout << "NO Camera or Just one,please check" << endl;
            return;
        }
        camera_car->read(img_src_car);
        if (img_src_car.empty()) {
            cout << "IMG IS EMPTY" << endl;
            return;
        }
        result = yolo_radar_trt::work(yolo, img_src_car);
        for (auto i: result) {
            std::cout << "label:" << i.label << std::endl;
            std::cout << "bbox:" << i.bbox << std::endl;
        }
#ifdef VIDEO
        imshow("dst_car", img_src1);
        waitKey(1);
#endif
#ifdef RECORD
        writer_car << img_src1;
#endif
    }
    cap_car.release();
    return;
}

int main(int argc, char **argv) {

#if RADAR_MODE == true
    radar();
#elif RADAR_MODE == false
    car();
#endif
    return 0;
}