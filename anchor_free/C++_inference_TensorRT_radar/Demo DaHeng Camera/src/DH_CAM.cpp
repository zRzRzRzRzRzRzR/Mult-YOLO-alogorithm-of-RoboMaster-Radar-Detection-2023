#include "DH_CAM.h"
#include "include/yolov8_radar.h"
using namespace std;
using namespace cv;
Mat img_src1;
Mat img_src2;
Camera *camera_left = nullptr;
Camera *camera_right = nullptr;
std::vector<yolo_radar_trt::Object> result_left;
std::vector<yolo_radar_trt::Object> result_right;

inline static std::string getCurrentTime() {
    std::time_t result = std::time(nullptr);
    std::string ret;
    ret.resize(64);
    int wsize = sprintf((char *) &ret[0], "%s", std::ctime(&result));
    ret.resize(wsize);
    return ret;
}

void radar() {
    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V8); //预处理
    string path_left = REORD_PATH + getCurrentTime() + "_left.avi";
    string path_right = REORD_PATH + getCurrentTime() + "_right.avi";
    camera_left = new DHCamera(LEFT_SN);
    camera_right = new DHCamera(RIGHT_SN);
    camera_left->init(0, 0, 1280, 1024, 6000, 6, false);
    camera_right->init(0, 0, 1280, 1024, 6000, 6, false);
    cv::VideoWriter writer_left(path_left, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 50, cv::Size(1280, 1024));
    cv::VideoWriter writer_right(path_right, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 50, cv::Size(1280, 1024));
    namedWindow("LEFT", WINDOW_NORMAL);
    resizeWindow("LEFT", 640, 512);
    namedWindow("RIGHT", WINDOW_NORMAL);
    resizeWindow("RIGHT", 640, 512);
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
#ifdef RECORD
        writer_left.write(img_src1);
        writer_right.write(img_src2);
#endif
#ifdef VIDEOS
        imshow("LEFT", img_src1);
        imshow("RIGHT", img_src2);
#endif
        result_left = yolo_radar_trt::work(yolo, img_src1);
        for (auto i: result_left) {
//            std::cout << "Find a target" << std::endl;
//            std::cout << "label:" << i.label << std::endl;
//            std::cout << "bbox_left:" << i.rect << std::endl;
        }

    }
    writer_right.release();
    writer_left.release();
    return;
}
int main(int argc, char **argv) {
    radar();
}

//#include <chrono>
//#include <iostream>
//#include <yaml-cpp/yaml.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudastereo.hpp>
//#include "yolov8_radar.h"
//#include "DH_CAM.h"
//
//using namespace std;
//
//void calcu_disparity_cuda_bm(cv::Mat &left, cv::Mat &right, cv::Mat &disparity) {
//    cv::Ptr<cv::cuda::StereoBM> bm = cv::cuda::createStereoBM(128,9);
//    bm->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
//    bm->setPreFilterSize(9);
//    bm->setPreFilterCap(31);
//    bm->setMinDisparity(16);
//    bm->setTextureThreshold(20);
//    bm->setUniquenessRatio(15);
//    bm->setSpeckleWindowSize(100);
//    bm->setSpeckleRange(32);
//
//    cv::Mat disparity_bm(left.size(), CV_16S);
//    cv::cuda::GpuMat cudaDisparityMap(left.size(), CV_16S);
//    cv::cuda::GpuMat cudaDrawColorDisparity(left.size(), CV_8UC4);
//    cv::cuda::GpuMat cudaLeftFrame, cudaRightFrame;
//    cudaLeftFrame.upload(left);
//    cudaRightFrame.upload(right);
//    bm->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);
//    cudaDisparityMap.download(disparity_bm);
//    imshow("df",disparity_bm);
//    disparity_bm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
//    auto t_b = chrono::high_resolution_clock::now();
//}
//void calcu_disparity_cpu_sgbm(cv::Mat &left, cv::Mat &right, cv::Mat &disparity) {
//    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
//    sgbm->setMinDisparity(16);
//    sgbm->setNumDisparities(16);
//    sgbm->setBlockSize(9);
//    sgbm->setP1(8 * left.channels() * 100);
//    sgbm->setP2(32 * left.channels() * 100);
//    sgbm->setPreFilterCap(63);
//    sgbm->setMinDisparity(0);
//    sgbm->setUniquenessRatio(50);
//    sgbm->setSpeckleWindowSize(100);
//    sgbm->setSpeckleRange(32);
//    sgbm->setDisp12MaxDiff(1);
//    sgbm->setMode(StereoSGBM::MODE_HH);
//
//    cv::Mat disparity_sgbm;
//    sgbm->compute(left, right, disparity_sgbm);
//    imshow("df",disparity_sgbm);
//    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
//
//}
//
//
//float calcu_distance(cv::Mat &disparity, cv::Rect bbox) {
//    cv::Mat roi_disp = disparity(bbox);
//    double minVal, maxVal;
//    cv::Point minLoc, maxLoc;
//    cv::minMaxLoc(roi_disp, &minVal, &maxVal, &minLoc, &maxLoc);
//    float distance = 0.0;
//    if (maxVal > 0.0) {
//        distance = 800 * 12.0 / (maxVal * 4); // 0.12为基线长度，721为焦距
//    }
//    return distance;
//}
//
//void radar() {
//    shared_ptr<yolo_radar_trt::Infer> yolo = yolo_radar_trt::prepare(yolo_radar_trt::Type::V8); //预处理
//    cv::Mat img_src1, img_src2;
//    Camera *camera_left = new DHCamera(LEFT_SN);
//    Camera *camera_right = new DHCamera(RIGHT_SN);
//    camera_left->init(0, 0, 1280, 1024, 10000, 6, false);
//    camera_right->init(0, 0, 1280, 1024, 10000, 6, false);
//
////    YAML::Node config = YAML::LoadFile("../camera.yaml");
////    cv::Mat K_left = cv::Mat(3, 3, CV_64FC1, config["camera_first"]["K"].as<std::vector<double>>().data());
////    cv::Mat D_left = cv::Mat(1, 5, CV_64FC1, config["camera_first"]["D"].as<std::vector<double>>().data());
////    cv::Mat K_right = cv::Mat(3, 3, CV_64FC1, config["camera_second"]["K"].as<std::vector<double>>().data());
////    cv::Mat D_right = cv::Mat(1, 5, CV_64FC1, config["camera_second"]["D"].as<std::vector<double>>().data());
////    cv::Mat R_left = cv::Mat(3, 3, CV_64FC1, config["camera_second"]["R"].as<std::vector<double>>().data());
////    cv::Mat R_right = cv::Mat(3, 3, CV_64FC1, config["camera_second"]["R"].as<std::vector<double>>().data());
////    cv::Mat P_left = cv::Mat(3, 4, CV_64FC1, config["camera_second"]["P"].as<std::vector<double>>().data());
////    cv::Mat P_right = cv::Mat(3, 4, CV_64FC1, config["camera_second"]["P"].as<std::vector<double>>().data());
////    cv::Mat R = cv::Mat(3, 3, CV_64FC1, config["camera_second"]["self_R"].as<std::vector<double>>().data());
////    cv::Mat T = cv::Mat(3, 1, CV_64FC1, config["camera_second"]["self_T"].as<std::vector<double>>().data());
////    cv::Size image_size(1280, 1080);
////    cv::Mat Q;
////    cv::stereoRectify(K_left, D_left, K_right, D_right, image_size, R, T, R_left, R_right, P_left, P_right, Q);
//
//    while (waitKey(1) != 27) {
//        if ((!camera_left->start()) || (!camera_right->start())) {
//            cout << "NO Camera or Just one,please check" << endl;
//            return;
//        }
//        camera_left->read(img_src1);
//        camera_right->read(img_src2);
//        if (img_src1.empty() || img_src2.empty()) {
//            cout << "IMG IS EMPTY" << endl;
//            return;
//        }
//
//        cv::Mat left_gray, right_gray;
//        cv::cvtColor(img_src1, left_gray, cv::COLOR_BGR2GRAY);
//        cv::cvtColor(img_src2, right_gray, cv::COLOR_BGR2GRAY);
//
//        cv::Mat disparity_cuda;
//        calcu_disparity_cuda_bm(left_gray, right_gray, disparity_cuda);
//
//        vector<yolo_radar_trt::Object> result = yolo_radar_trt::work(yolo, img_src1);
//        for (auto i : result) {
//            float distance = calcu_distance(disparity_cuda, i.rect);
//            std::cout << "distance:" << distance << std::endl;
//        }
//        cv::imshow("LEFT", img_src1);
////        cv::imshow("Disparity", disparity_cuda);
//    }
//
//    delete camera_left;
//    delete camera_right;
//    return;
//}
//
//int main(int argc, char **argv) {
//    radar();
//    return 0;
//}