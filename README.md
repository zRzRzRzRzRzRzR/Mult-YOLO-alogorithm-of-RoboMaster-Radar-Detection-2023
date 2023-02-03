**2023 RoboMaster XJTLU Radar Object Detection**
=

### **Team: 动云科技GMaster战队 <br>**

#### **Author: *视觉组 张昱轩 zR***

## 功能介绍

动云科技GMaster战队2023赛季 yolo目标检测 雷达小目标模型训练代码。<br>
本仓库包含从数据集制作到推理部署全套代码。并分为Anchor_free 和 Anchor_base两种检测头(yolov7,yolov8)
机制，其对应的文件夹分别为 ```anchor_free```和```anchor_base```<br>
由于算法的不同，两种算法的训练，后处理方式和推理方式是不同的，故两份代码中将会包含以下文件。

```
yolo_anchorfree/anchorbase_rader: 检测网络训练框架
C++_inference_Openvino_radar(Only Anchor_base): OpenVINO推理代码（用于Inter的CPU和GPU推理加速)
C++_inference_TensorRT_radar: TensorRT推理代码（用于NVIDIA的GPU推理加速)
```

**```注意```**

+ 由于团队在2022年11月已经弃用OpenVINO推理加速。故该部分代码不会继续维护。同时，也不在提供Anchor_free版本的OpenVINO推理代码，如有需要，可以参考以下方式进行修改：
  + [YOLOX ](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/OpenVINO)OpenVINO推理模型
  + __2023 RoboMaster XJTLU Armor Keypoints Detection__ 关键点模型OpenVINO推理代码(即将开源)。
+ 请仔细阅读每份代码的```README.md```文件。
+ 如果你需要下载最新版本的代码，请克隆```dev```分支，最新的代码无法保证性能稳定。

## 环境配置

我们团队的训练配置和推理配置如下
***
|硬件设备| 训练设备|推理设备|
| - | - | - |
| CPU | 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz | 12th Gen Intel i7-12700 |
| GPU | NVIDA A5000 x 8 | NVIDA RTX3060 12GB / Inter A750 |
| Memory| 80GB | 16GB |

| 环境配置 |训练设备 |推理设备 |
| - | - | - |
| OS | Ubuntu 20.04.5 | Ubuntu 20.04.5 |
| CUDA | 11.3 | 11.8 |
|Kernel | 5.15.0 | 5.15.0 |
|gcc/g++ | 9.4.0 | 9.4.0 |
|cmake | \ | 3.16.3 |
|Python | 3.8.6 | 3.10.8 |
|ONNX | 1.13.0 | 1.13.0 |
|Pytorch | 1.11.0 | \ |

**```注意```**

+ 更详细的推理设备配置文件以及注意事项，请查看对应功能的```README.md```文件。

## 数据集<br>

### 数据集格式和标注方法

我们团队使用 [labelimg](https://github.com/heartexlabs/labelImg) 工具进行标注并保存为yolo格式。<br>
本代码基于 [YoloV7](https://github.com/WongKinYiu/yolov7) 框架开发，理论上支持COCO类型数据集。

### 数据集信息

#### 数据集来源:<br>

+ 西交利物浦大学RM2023赛季场地数据集数据集录制 2500张。
+ 共计 2500张。
  <br><br>

#### 数据集分配:<br>

+ 数据共有14类，分别为

***
|编号 | 含义 | 序号 |编号 | 含义 | 序号 |
|-|-|-|-|-|-|
| B1 | 蓝方一号装甲板 | 0 | R1 | 红方一号装甲板 | 7 |
| B2 | 蓝方二号装甲板 | 1 | R2 | 红方二号装甲板 | 8 |
| B3 | 蓝方三号装甲板 | 2 | R3 | 红方三号装甲板 | 9 |
| B4 | 蓝方四号装甲板 | 3 | R4 | 红方四号装甲板 | 10 |
| B5 | 蓝方五号装甲板 | 4 | R5 | 红方五号装甲板 | 11 |
| BO | 蓝方前哨站装甲板 | 5 | RO| 红方前哨站装甲板 | 12 |
| BS | 蓝方哨兵装甲板 | 6 | RS | 红方哨兵装甲板 | 13 |

+ 如果该数据集类别顺序与你的不相符， 你可以使用```pre-processing_script/change_change_label.py```
  脚本批量修改你的标签。或者修改代码成符合你的数据集顺序。
+ 数据集按照 8:1:1的比例分配为训练集，验证集和测试集。
  超参数设置:<br>
+ 训练超参数未对yolov7/yolov8原本的超参数进行过多的调整，主要调整了数据集增强的部分。关闭了上下，左右的反转。同时修改了一些其他数据预处理参数。

## 训练流程<br>

### Train:

#### Single GPU:

```python train.py --workers 8 --device 0 --batch-size 8 --data data/armor-radar.yaml --img 640 640 --cfg cfg/GMaster/yolov7_4anchor.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.tiny.yaml ```

#### Muti GPUs:

``` python -m torch.distributed.launch --nproc_per_node 8 train.py --workers 64 --batch-size 64```

### Inference:

#### On Videos and Image:

```python detect.py --weights yolov7_armor.pt --conf 0.25 --img-size 640 --source yourvideo.mp4```


### Export:

- Pytorch to ONNX:<br>
    - 导出方法与[YoloV7](https://github.com/WongKinYiu/yolov7)官方方法相同，如果部署在TensorRT上，建议使用Deploy的训练方式，便于导出。
- ONNX to OpenVINO:<br>
    - 将ONNX文件转换为OpenVINO文件 仅需要执行以下步骤:<br>
    - 在[安装OpenVINO](https://docs.openvino.ai/latest/OpenVINO_docs_install_guides_overview.html)
      的Python环境下，执行<br>
      ```mo --input_model /path/to/your_models.onnx --output_dir /path/to/out_dir```
    - 将转换得到的xml, bin, mapping 文件放置在一个文件夹，接入推理代码。
    - 建议使用 [ONNX version](https://github.com/onnx/onnx) == 13.0 作为ONNX中间导出版本。
- ONNX to TensorRT:<br>
    - 在[安装TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
      的推理设备上将onnx文件转换为引擎文件，仅需执行以下步骤:<br>
      ```/usr/src/tensorrt/bin/trtexec --onnx=your_models.onnx --saveEngine=your_models.FP32.trtmodel```<br>
      FP16,INT8的方法测试可用，均可使用TensorRT自带的工具实现。
    - 将转换后的引擎文件接入推理代码。
    - 建议使用 ONNX version >= 15.0 <=17.0 作为ONNX中间导出版本。
- Pytorch to TenosrRT & OpenVINO :<br>
    - 本算法框架是继承Yolov7的算法框架并进行二次开发，理论上可以直接从Pytorch直接导出至TensorRT或OpenVINO引擎。
    - 由于我们团队的训练和推理不在一台机器上，故没有进行相关方面的尝试，使用者可自行尝试。

### 效果展示

![](https://github.com/zRzRzRzRzRzRzR/Mult-YOLO-alogorithm-of-RoboMaster-Radar-Detection-2023/blob/main/show_pic/inference.jpg)

### 部署和推理

**```注意```**

+ anchor_free和anchor_base的推理代码并不相同。但在本份代码中，我们团队都使用了近似的命名方式和解构，便于使用者理解。
+ 使用前请检查硬件和系统配置是否满足要求。

#### Inference in OpenVINO (x86):

+
若使用anchor_base进行推理，工程中设定的anchor需要根据实际的应用场景进行人工调整，anchor修改请执行```pre-processing_script/change_anchor.py```
。
+ OpenVINO推理代码位于文件夹 ```C++_inference_OpenVINO_radar``` 中，仅做简单推理，请根据需求自行接入工程。
+ 更多使用说明，请查看```C++_inference_OpenVINO_radar```中的```README.md```文件。

### Inference in TensorRT (arm & x86):

+ OpenVINO推理代码位于文件夹 ```C++_inference_TensorRT_radar``` 中，仅做简单推理，请根据需求自行接入工程。
+ anchor信息会根根据导出的anchor信息锁定，无法修改，请提前人工设定好anchor。
+ 更多使用说明，请查看```C++_inference_TensorRT_radar```中的```README.md```文件。

## 总结

+ 该项目针对RMUC2023赛季，如果你有好的建议，欢迎给我留言哦。
+ 如果你觉得项目对你有帮助，请给个star吧~
+ 如果你有更好的建议，欢迎提出PR，或者直接联系我哦，大家一起学习！

## 联系方式

+ 作者微信: zR_ZYX
+ 作者邮箱: Yuxuan.Zhang2104@student.xjtlu.edu.cn
+ 团队邮箱: TeamGMaster@xjtlu.edu.cn