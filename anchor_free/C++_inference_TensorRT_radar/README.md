**2022-2023 RoboMaster XJTLU Radar Detection(C++ TensorRT)**
==

### **Team: 动云科技GMaster战队 <br>**

#### **Author: *视觉组 张昱轩 zR***

***

## 功能介绍

本代码用于将视觉双目相机雷达的anchor_free(基于yolov8算法)权重文件部署至 C++ TensorRT环境。<br>

## DEMO演示代码架构

├── CMakeLists.txt<br>
├── demo_radar.cpp <br>
├── yolov8_raadr.cu <br>
├── yolov8_raadr.cpp <br>
└── yolov8_radar.h <br>


### Demo DaHeng Camera

该目录下包含了一个简单的大恒水星相机的调用程序，并引入了目标检测实现打开摄像头的检测。<br>

## 代码解释和使用说明:

### demo_radar.cpp:<br>

主代码，将你测试的视频的绝对路径填写至 ```VIDEO_PATH``` 即可。<br>

### yolov8_radar.h:<br>

包含了头文件和所有的宏定义函数，其中对应的含义如下:<br>

+ ```MODEL_PATH```推理网络模型位置，在这里放入 ```.trtmodel``` 或 ```.engine```量化权重文件。<br>
  ```注意```: 请将群众文件放在DEMO的主目录下，使用相对位置书写。以下为一种书写的案例:

```
#define MODEL_PATH "../armor.FP16.trtmodel" 
```

+ ```DEBUG``` 调试模式，默认不启动。如果使用则取消注释。<br>
  在```DEBUG```模式下，会显示更加详细的推理状态以及每一帧的推理时间等。
+ ```VIDEOS``` 是否显示推理视频，默认不启动。如果使用则取消注释。
+ ```NMS_THRESHOLD``` nms阈值，默认值 ```0.45```。
+ ```CONF_THRESHOLD``` 置信度阈值，默认值 ```0.5```。
+ ```GPU_ID``` 推理设备，默认值为```0```。
+ ```注意```:
  + TensorRT部署不需要手动配置anchor数据，因此，采用TensorRT部署不需要像Openvino手动计算anchor的数量和聚类数据。
  + TensorRT仅能在NVIDIA的GPU上推理，固设备无法选择CPU。

部分类，函数的解释如下:<br>

+ ```yolo_radar_trt``` : ```yolov8_radar.cu```的命名空间。
+ ```Type```: 推理的类型，目前仅写入了yolov8，V5版本可以按照v8的方式推理。 若之后的yolo版本推理方式类似，可以直接加入。
+ ```Mode```: 推理模式，支持INT8，FP16，FP32，其中INT8需要标定数据。本代码没有提供ONNX转TensoRT的INT8方案。
+ ```Object```,```Object_result```: 结果类型，详见```API接口```部分。
+ ```Infer```: 推理类，生成的TensorRT推理类，用于推理。

### yolov8_radar.cu:<br>

该部分主要是推理部分的cuda代码复现和定制算子。部分类的解释如下:<br>
```MonopolyAllocator``` 独占管理类，通过对tensor做独占管理，具有max_batch * 2个tensor，通过query获取一个
当推理结束后，该tensor释放使用权，即可交给下一个图像使用，达到内存复用的目的。<br>
```ThreadSafedAsyncInfer```
异步线程安全推理，通过异步线程启动，使得调用方允许任意线程调用把图像做输入，并通过future来获取异步结果。<br>
```YoloTRTInferImpl```
Yolov8的具体实现。实现预处理的计算重叠、异步垮线程调用，最终拼接为多个图为一个batch进行推理。最大化的利用显卡性能，实现高性能高可用好用的yolo推理。<br>

```Transpose_kernel```
将Yolov8的输出的[1, num_class, info]形式转换为[1, info, num_class]

### yolov8_radar.cpp:<br>

包含了所有的推理代码，部分函数的解释如下:<br>

+ ```work``` 推理主函数。
+ ```drawPred```: 绘制预测框和预测信息。

#### API接口:<br>

返回值类型为自定义的```Object_result```结构体，包含如下信息:<br>
struct Object_result<br>
├── ```<int> label``` 识别的标签类别。<br>
├── ```<float> prob```  识别的置信度。<br>
└── ```<rect> bbox``` 识别的bbox信息，包含左上角的坐标和宽高<br>

## 推理样例展示:

图片展示:<br><br><br>
![](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/show_pic/demo_trt.png)<br>
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
GPU: RTX3060 12GB 
Memory: 32GB
```

```
系统配置:
OS: Ubuntu 20.04.5
Kernel: 5.15.0
Cuda  : 11.8 
Cudnn : 8.7 
TensorRT : 8.5GA
gcc/g++ : 9.4.0
cmake : 3.16.3
Python : 3.10.6
ONNX : 15
OpenCV : 4.6.0
Protobuf : 3.11.4
Protobuf : 3.11.4
```
__注意: 该代码在上述指定版本以下 (尤其是与显卡相关的库) 的版本可能无法运行。 例如在```CUDA 10.2```（以及对应的TensorRT，Cudnn版本)下可能会无法正确读取网络模型。__