#include "include/depth_radar.h"

std::vector<yolo_radar_trt::Object> left_result;
std::vector<yolo_radar_trt::Object> right_result;
double compensation_factor = 0;
int frame_counter = 0;
int limit = 10;
double compensation_sum = 0.0;

DepthRadar radar;

cv::Point DepthRadar::getCenterPoint(yolo_radar_trt::Object res) {
    return cv::Point(res.rect.x + res.rect.width / 2, res.rect.y + res.rect.height / 2);
}

vector<yolo_radar_trt::Object>
DepthRadar::processFrames(shared_ptr<yolo_radar_trt::Infer> yolo, cv::Mat left_frame, cv::Mat right_frame) {
    left_result = yolo_radar_trt::work(yolo, left_frame, "left_radar");
    right_result = yolo_radar_trt::work(yolo, right_frame, "right_radar");
    double group1_real_size = 82; // Group1的真实尺寸，例如135mm
    double group2_real_size = 57; // Group2的真实尺寸，例如82mm
    double group_real_size = 110; // Group的真实尺寸，例如82mm
    // Calculate compensation factor
    if (frame_counter < limit) {
        double group1_compensation = 0;
        double group2_compensation = 0;
        double group1_height = 2761;
        double group2_height = 3307;
        for (int i = 0; i < left_result.size(); i++) {
            for (int j = 0; j < right_result.size(); j++) {
                if (left_result[i].label == right_result[j].label &&
                    (left_result[i].label == 5 || left_result[i].label == 12)) {
                    double disparity = abs(getCenterPoint(left_result[i]).x - getCenterPoint(right_result[j]).x);
                    double calculated_depth = (BASELINE * FOCAL_LENGTH) / (disparity * PIXEL_SIZE);

                    // 计算像素尺寸，这里我们使用物体的宽度
                    double pixel_size = left_result[i].rect.width;
                    // 计算真实尺寸与像素尺寸的比例
                    double scale_factor = group1_real_size / pixel_size;
                    // 使用比例因子来调整深度估计
                    calculated_depth *= scale_factor;

                    cout<<"calculated_depth1:"<<calculated_depth<<endl;
                    group1_compensation += KNOWN_DEPTH1 / calculated_depth;
                } else if (left_result[i].label == right_result[j].label &&
                           (left_result[i].label == 14 || left_result[i].label == 15)) {
                    double disparity = abs(getCenterPoint(left_result[i]).x - getCenterPoint(right_result[j]).x);
                    double calculated_depth = (BASELINE * FOCAL_LENGTH) / (disparity * PIXEL_SIZE);

                    // 计算像素尺寸，这里我们使用物体的宽度
                    double pixel_size = left_result[i].rect.width;
                    // 计算真实尺寸与像素尺寸的比例
                    double scale_factor = group2_real_size / pixel_size;
                    // 使用比例因子来调整深度估计
                    calculated_depth *= scale_factor;
                    cout<<"calculated_depth2:"<<calculated_depth<<endl;
                    group2_compensation += KNOWN_DEPTH2 / calculated_depth;
                }
            }
        }
        cout << "frame:" << frame_counter << " group1_compensation:" << group1_compensation << " group2_compensation"
             << group2_compensation << endl;
        if (group1_compensation == 0 || group2_compensation == 0) {
            cout << "lossing detect one of base or Ord,Please check" << endl;
            limit++;
        } else
            compensation_sum += (group1_compensation + group2_compensation) / 2;
        frame_counter++;
        if (frame_counter >= limit) {
            compensation_factor = compensation_sum / 10.0;
            cout << "compensation finish: " << compensation_factor << ", Use total frame: " << limit << endl;
            cout << "Press any key to continue...";
            cin.get();
        }
    }
    if (frame_counter >= 10) {
// Calculate depth
        for (int i = 0; i < left_result.size(); i++) {
            for (int j = 0; j < right_result.size(); j++) {
                if (left_result[i].label == right_result[j].label &&
                    !(left_result[i].label == 5 || left_result[i].label == 12 ||
                      left_result[i].label == 14 || left_result[i].label == 15)) {
                    double disparity = abs(getCenterPoint(left_result[i]).x - getCenterPoint(right_result[j]).x);
                    left_result[i].depth = (BASELINE * FOCAL_LENGTH) / (disparity * PIXEL_SIZE) * compensation_factor;
                    double pixel_size = left_result[i].rect.width;
                    // 计算真实尺寸与像素尺寸的比例
                    double scale_factor = group_real_size / pixel_size;
                    left_result[i].depth *= scale_factor;
                }
            }
        }
    }
// 删除深度为0的对象
    left_result.erase(std::remove_if(left_result.begin(), left_result.end(),[](const yolo_radar_trt::Object &obj) { return obj.depth == 0; }),left_result.end());
    return left_result;
}
