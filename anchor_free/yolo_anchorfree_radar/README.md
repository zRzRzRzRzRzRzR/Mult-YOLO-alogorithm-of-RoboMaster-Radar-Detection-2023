***2023 RoboMaster XJTLU Radar Object Detection Anchor_Free training code***
=
### **Team: 动云科技GMaster战队 <br>**
#### **Author: *视觉组 张昱轩 zR***
***
## 功能介绍

本代码使用anchor_free的方法训练RoboMaster雷达小目标检测网络。<br>
主干网络的优化方式与anchor_base的方法相同，主要在于更改检测头和后处理的方式。<br>
本代码基于[YoloV7](https://github.com/WongKinYiu/yolov7)框架书写。部分源码和使用方法也可参考官方代码库。<br>

### 测试过的网络和表现

+ 推理代码详见[OpenVino_Demo](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/C%2B%2B_inference_Openvino_radar)
  和[TensorRT_Demo](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/C%2B%2B_inference_TensorRT_radar)
+ 推理显卡为RTX 3060，更多硬件参数详见``` Demo```介绍。
+ 本表格中的所有的推理均在 FP16 精度下进行。控制IOU参数为``` 0.6```, NMS参数为``` 0.4 ```。
+ 若一个网络无法检测到小目标，则之后的测试项目将没有必要测试。
+ 表格中的符号表示如下：
  + ``` √ ``` 达到了团队要求的比赛的标准（上述要求精度）
  + ``` × ``` 无法推理 / 精度不达标 / 速度不达标。
  + ``` \ ``` 符号表表示 未测试 / 无法测试 / 没有测试必要（由于队伍的推理框架由Openvino转向了TensorRT，大部分算法实验不在测试Openvino独显性能)
  + ``` ? ``` 未测试。
  + ```[]```  替代标准  yolov8 模块的位置 ，例如 ``` [3]```代表将第三个标准C2f模块更换为指定模块。

#### Performance
由于yolov8的性能表现目前暂时没有团队修改过的yolov7效果更好，

|       主干网络        | 激活函数 | IOU   | Head            |  训练轮次 |  小目标 | mAP@.5| mAP@.5:.95 | 参数量   |   TRT FPS  |
|-|-|-|-|-|-|-|-|-|-|
| C2f                 | SiLU    | GIOU | C2f              |   150    |    ×   |  0.893 |   0.638   |  165MB |    55    |  
| C2f       + 4detlay | SiLU    | GIOU | C2f + SimAM      |   150    |    √   |  0.901 |   0.691   |  153MB |    25    |
| C2f       + 4detlay | FRelu   | SIOU | CoT2f + SimAM    |   150    |    √   |  0.901 |   0.603   |  155MB |    29    |
