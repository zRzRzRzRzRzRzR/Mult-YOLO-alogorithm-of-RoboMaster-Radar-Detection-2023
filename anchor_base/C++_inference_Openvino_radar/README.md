**2022-2023 RoboMaster XJTLU Radar Detection(C++ Openvino)**
=

### **Team: 动云科技GMaster战队 <br>**

#### **Author: *视觉组 张昱轩 zR***

*** 

## 功能介绍

本代码用于将视觉双目相机雷达的anchor_base网络权重文件部署至 C++ Openvino环境。<br>

**```注意```** 
+ 由于团队在2022年11月已经弃用OpenVINO推理加速。固该部分代码不会继续维护。

## DEMO演示代码架构

├── CMakeLists.txt<br>
├── demo_radar.cpp <br>
├── yolov7_raadr.cpp <br>
└── yolov7_radar.h  <br>

## 代码解释和使用说明：

### demo_radar.cpp:<br>

主代码，将你测试的视频的绝对路径填写至 ```VIDEO_PATH``` 即可。<br>

### yolov7_radar.h：<br>

包含了头文件和所有的宏定义函数，其中对应的含义如下：<br>

+ ```MODEL_PATH```推理网络模型位置，在这里放入 ```.xml``` 或 ```.onnx```量化权重文件。
+ ```DEBUG``` 调试模式，默认不启动。如果使用则取消注释。<br>
  在```DEBUG```模式下，会显示更加详细的推理状态以及每一帧的推理时间等。
+ ```VIDEOS``` 是否显示推理视频，默认不启动。如果使用则取消注释。
+ ```IMG_SIZE``` 图像推理大小，取决与你的模型，并需要修改ANCHOR的具体值。默认值为```416```。
+ ```NMS_THRESHOLD``` nms阈值，默认值 ```0.2```。
+ ```CONF_THRESHOLD``` 置信度阈值，默认值 ```0.7```。
+ ```ANCHOR``` ANCHOR的数量，默认值为```3```。
+ ```ANCHOR_SMALL``` 添加小目标ANCHOR的数量，默认值为```0```。<br>
  ```ANCHOR_SMALL```最多允许设定为2，即添加了P2,P1两层检测层。当其设为```1```时，默认添加P2检测层。
+ ```ANCHOR_BIG``` 添加大目标ANCHOR的数量，默认值为```0```。
+ ```ANCHOR_SMALL```最多允许设定为2，即添加了P6 P7两层检测层。当其设为```1```时，默认添加P6检测层。
+ ```CLS_NUM```分类的数量，默认值为团队所使用的```14```类。
+ ```DEVICE``` 推理设备，默认值为```CPU```。

### yolov7_radar.cpp：<br>

+ ```letter_box```,```scale_box```,```generate_proposals```: yolov7推理中对应的部分。
+ ```drawPred```: 绘制预测框和预测信息。
+ ```work``` : 推理主函数。

#### API接口:<br>

返回值类型为自定义的```Object_result```结构体，包含如下信息:<br>
struct Object_result<br>
├── ```<int> label``` 识别的标签类别。<br>
├── ```<float> prob```  识别的置信度。<br>
└── ```<rect> bbox``` 识别的bbox信息，包含左上角的坐标和宽高<br>

## 推理样例展示：

图片展示:<br><br><br>
![](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/show_pic/demo_openvino.png)<br>

输出展示：<br><br>
单目车载相机:
```
label:4
bbox:[54 x 42 from (2118, 846)]
Time: 7.082471
```
双目雷达相机:
```
Find a target
label:1
bbox_left:[105 x 139 from (389, 593)]
bbox_right:[106 x 136 from (555, 579)]

```
## 推理代码测试平台

```
硬件设备:
CPU: 12th Gen Intel i7-12700 
GPU: A750 8GB 
Memory: 32GB
```

```
系统配置:
OS: Windows 11 22H2 / Ubuntu 20.04.5
Kernel: 5.15.0
Openvino: 2022.1
gcc/g++ : 9.4.0
cmake : 3.16.3
Python : 3.8.5
ONNX : 15
OpenCV : 4.6.0
```

__注意: 该代码在上述指定版本以下的版本可能无法运行。 例如在```OpenCV 4.2.2```下没有```cv::dnn::SoftNMS```
方法,请更换为```cv::dnn::NMS```。__