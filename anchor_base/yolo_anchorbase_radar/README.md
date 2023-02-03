***2023 RoboMaster XJTLU Radar Object Detection Anchor_Base training code***
=
### **Team: 动云科技GMaster战队 <br>**
#### **Author: *视觉组 张昱轩 zR***
***
## 功能介绍

本代码使用anchor_base的方法训练RoboMaster雷达小目标检测网络。<br>
主干网络的优化方式与anchor_base的方法相同，主要在于更改检测头和后处理的方式。<br>
本代码基于[YoloV7](https://github.com/WongKinYiu/yolov7)框架书写。部分源码和使用方法也可参考官方代码库。<br>

### 测试过的网络和表现

+ 推理代码详见[OpenVino_Demo](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/tree/main/anchor_base/C%2B%2B_inference_Openvino_radar)
  和[TensorRT_Demo](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/tree/main/anchor_base/C%2B%2B_inference_TensorRT_radar)
+ 推理显卡分别为Inter A750 和 RTX 3060，更多硬件参数详见``` Demo```介绍。
+ 本表格中的所有的推理均在 FP16 精度下进行。控制IOU参数为``` 0.6```, NMS参数为``` 0.4 ```。
+ 若一个网络无法检测到小目标，则之后的测试项目将没有必要测试。
+ 表格中的符号表示如下：
  + ``` √ ``` 达到了团队要求的比赛的标准（上述要求精度）
  + ``` × ``` 无法推理 / 精度不达标 / 速度不达标。
  + ``` \ ``` 符号表表示 未测试 / 无法测试 / 没有测试必要（由于队伍的推理框架由Openvino转向了TensorRT，大部分算法实验不在测试Openvino独显性能)
  + ``` ? ``` 未测试。
  + ```[]```  替代标准 yolov7/yolov8 模块的位置 ，例如 ``` [3]```代表将第三个标准ELAN模块更换为指定模块。

#### Performance

|       主干网络        | 激活函数 | IOU   | Head            |  训练轮次 |  小目标 | mAP@.5| mAP@.5:.95 | 参数量 | TRT FPS | OpenVino FPS |
|-|-|-|-|-|-|-|-|-|-|-|
| ELAN                | SiLU    | GIOU | ELAN             |   100   |    ×   |   ×   |   ×   |  78MB  |   120   |      69      |
| ELAN                | SiLU    | GIOU | STD[3]           |   100   |    ×   |   ×   |   ×   |  82MB  |   102   |      41      |
| ELAN                | SiLU    | GIOU | STD              |   100   |    ×   | 0.512 | 0.302 |  80MB  |   102   |      41      |
| Swin-TransformerV2  | SiLU    | GIOU | C3STR            |   100   |    ×   |   ×   |   ×   |   ×    |    ×    |       ×      |
| CH3B[2,4]  + 4detlay| SiLU    | GIOU | BOT[2,3]         |   100   |    √   | 0.611 | 0.429 |  142MB |    61   |       ×      |
| CH3B[2,4]  + 4detlay| SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.432 | 0.391 |  143MB |    60   |       ×      |
| CH3B      + 4detlay | SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.65  | 0.427 |  141MB |    ?    |       ×      |
| CH3B[2,4] + 4detlay | SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.479 | 0.309 |  142MB |    58   |       \      |
| CH3B + 4detlay      | SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.624 | 0.405 |  142MB |    49   |       \      |
| CNeB[2,4] + 4detlay | SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.520 | 0.332 |  150MB |    52   |      27      |
| CNeB     + 4detlay  | SiLU    | SIOU | COT + SimAM      |   100   |    √   | 0.555 | 0.333 |  150MB |    52   |      27      |
| CNeBV2   + 4detlay  | FReLU   | SIOU | COT + SimAM      |   100   |    √   | 0.571 | 0.390 |  141MB |    50   |      27      |
| CNeBV2   + 4detlay  | FReLU   | SIOU | COT + SimAM      |   100   |    √   | 0.889 | 0.674 |  141MB |    50   |      27      |
| CH3B + 4detlay      | FReLU   | SIOU | COT + SimAM      |   100   |    √   | 0.819 | 0.557 |  176MB |    47   |      26      |
| CH3B + 4detlay      | FReLU   | SIOU | COT + SimAM      |   200   |    √   | 0.952 | 0.699 |  176MB |    47   |      26      |

### 最终结构
我们团队最终使用的网络模型配置文件为 ```cfg/GMaster/yolov7-4anchor-transCoT3-Full-HorNet-SimAM.yaml ```。<br>
下图为该网络的简单结构图：<br>
![](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/show_pic/yolo.png)