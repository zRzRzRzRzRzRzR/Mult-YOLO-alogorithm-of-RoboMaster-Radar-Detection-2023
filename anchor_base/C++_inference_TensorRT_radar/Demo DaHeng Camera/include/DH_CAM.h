#ifndef CAMERA_DH_CAM_H
#define CAMERA_DH_CAM_H

#include "CamWrapper.h"
#include "CamWrapperDH.h"

#define REORD_PATH  "../" //视频录制的位置
#define RADAR_MODE true //是否开启雷达模式，若关闭则为机器人车载模式
//#define RECORD_1 //是否保存视频
#define LEFT_SN "FGV22100003" //雷达左摄像头
#define RIGHT_SN "FGV22100004" //雷达右摄像头
#define CAR_SN "FGV22100003" //车载摄像头

//#define CAR_SN "KN0210060029"
//#define CAR_SN "KN0210060030"
//#define CAR_SN "KE0170050050"
inline static std::string getCurrentTime();

void radar();

void car();

#endif //CAMERA_DH_CAM_H