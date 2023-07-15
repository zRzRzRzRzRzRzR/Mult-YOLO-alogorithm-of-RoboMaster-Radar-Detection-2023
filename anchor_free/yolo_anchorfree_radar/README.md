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

+

推理代码详见[TensorRT_Demo](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/tree/main/anchor_free/C%2B%2B_inference_Openvino_radar)

+ 推理显卡为RTX 3060，更多硬件参数详见``` Demo```介绍。
+ 本表格中的所有的推理均在 FP16 精度下进行。控制IOU参数为``` 0.6```, NMS参数为``` 0.4 ```。
+ 若一个网络无法检测到小目标，则之后的测试项目将没有必要测试。
+ 表格中的符号表示如下：
    + ``` √ ``` 达到了团队要求的比赛的标准（上述要求精度）
    + ``` × ``` 无法推理 / 精度不达标 / 速度不达标。
    + ``` \ ``` 符号表表示 未测试 / 无法测试 /
      没有测试必要（由于队伍的推理框架由Openvino转向了TensorRT，大部分算法实验不在测试Openvino独显性能)
    + ``` ? ``` 未测试。
    + ```[]```  替代标准 yolov8 模块的位置 ，例如 ``` [3]```代表将第三个标准C2f模块更换为指定模块。

#### Performance

我们团队测试的主干网络以YOLOv7为主，请查看[这里](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/tree/dev/anchor_base/yolo_anchorbase_radar)
获取更多我们做过的测试。<br>
在算法改进中，我们团队发现，采用YOLOv8s作为预训练模型已足够满足竞赛需求，为了实现更快的帧率，我们使用了YOLOv8s作为基线模型，固以下测试中的默认配置为YOLOv8s的配置。
| 主干网络 | 激活函数 | IOU | Head | 训练轮次 | 小目标 | mAP@.5| mAP@.5:.95 | 参数量 | TRT FPS |
|-|-|-|-|-|-|-|-|-|-|
| C2f(YOLV8_OFFICIAL) | SiLU | GIOU | C2f | 300 | × | 0.913 | 0.638 | 72MB | 90 |  
| C2f + 4detlay | SiLU | GIOU | C2f | 300 | √ | 0.963 | 0.658 | 79MB | 65 |
| C2fHB + 4detlay | SiLU | GIOU | CoT2f + SimAM | 300 | √ | 0.988 | 0.751 | 70MB | 50 |

### 最终结构

我们团队最终使用的网络模型配置文件为 ```cfg/GMaster/yolov8_4_C2fHB_COT2f_SimAM.yaml ```。<br>
下图为该网络的简单结构图：<br>
![](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/show_pic/v8.jpg)
