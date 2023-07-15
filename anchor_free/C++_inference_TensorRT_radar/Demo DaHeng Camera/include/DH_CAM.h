#ifndef CAMERA_DH_CAM_H
#define CAMERA_DH_CAM_H

#include "CamWrapper.h"
#include "CamWrapperDH.h"

#define RECORD
#define REORD_PATH  "/home/zr/record_radar/" //视频录制的位置
#define LEFT_SN "FGV22100003" //雷达左摄像头
#define RIGHT_SN "FGV22100004" //雷达右摄像头
inline static std::string getCurrentTime();
void radar();
#endif //CAMERA_DH_CAM_H