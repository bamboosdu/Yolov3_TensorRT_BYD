<!--
 * @Author: your name
 * @Date: 2020-10-26 17:59:59
 * @LastEditTime: 2025-04-02 23:52:09
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Description: In User Settings Edit
 * @FilePath: /yolov3_tensorrt/README.md
-->

# YOLOv3 TensorRT 目标检测项目

这是一个基于YOLOv3和TensorRT的目标检测项目，支持实时视频检测和目标跟踪。
![yolo&yolo+sort](./sort.gif)

## 项目结构

```
.
├── common/          # 通用工具函数
├── deep_sort/       # DeepSORT目标跟踪算法实现
├── eval_yolo.py     # YOLO模型评估脚本
├── plugins/         # TensorRT插件相关文件
├── utils/          # 工具函数(可视化、预处理等)
├── yolo/           # YOLO模型相关文件
├── trt_yolo.py     # 基础目标检测实现
├── trt_yolo_with_screen.py  # 带目标跟踪的检测实现
└── setup.py        # 项目安装配置
```

## 功能特点

- 支持YOLOv3目标检测
- 使用TensorRT加速推理
- 集成DeepSORT目标跟踪
- 支持实时视频流处理
- 支持图片和视频输入
- 可视化检测和跟踪结果

## 环境要求

- Python 2.7
- CUDA
- TensorRT 7.1.3.4
- OpenCV
- PyCUDA
- ONNX 1.4.1

## 安装步骤

1. 安装依赖包:
```bash
# 安装protobuf
bash install_protobuf-3.8.0.sh

# 安装pycuda
pip install pycuda==2019.1.1

# 安装onnx
sudo pip3 install onnx==1.4.1
```

2. 编译项目:
```bash
cd plugins
make
```

3. 生成TensorRT, 加速推理:
```bash
cd yolo
bash darknet2onnx.sh
bash onnx2trt.sh
```

## 使用方法

1. 基础目标检测:
```bash
python3 trt_yolo.py --image <image_path> -m yolov3-416
```

2. 视频检测(带目标跟踪):
```bash
python3 trt_yolo_with_screen.py --video <video_path> -m yolov3-416
```

参数说明:
- `-m/--model`: 模型名称，如yolov3-416
- `-c/--category_num`: 目标类别数量
- `--image`: 输入图片路径
- `--video`: 输入视频路径

## 注意事项

1. 确保TensorRT引擎文件(.trt)已正确生成
2. 视频处理时可按ESC键退出
3. 按F键切换全屏显示
4. 检测结果将保存为result_3.avi

## 许可证

MIT License



