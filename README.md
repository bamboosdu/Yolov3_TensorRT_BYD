<!--
 * @Author: your name
 * @Date: 2020-10-26 17:59:59
 * @LastEditTime: 2020-10-26 18:10:13
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /yolov3_tensorrt/README.md
-->
```
.
├── common
├── deep_sort
├── eval_yolo.py
├── LICENSE
├── Makefile
├── plugins
├── __pycache__
├── pytrt.cpp
├── pytrt.pxd
├── pytrt.pyx
├── README.md
├── result_3.avi
├── run_yolo.sh
├── setup.py
├── trtNet.cpp
├── trtNet.h
├── trt_yolo.py
├── trt_yolo_with_screen.py
├── utils
└── yolo
```
![yolo&yolo+sort](./sort.gif)

The **deep_sort** file contains sort&deep_sort realization.

The **plugins** contains files about yolo net.

The **utils** contains files for visualization, preprocessing,etc.

The **yolo** is the most important one, it explains how to convert darknet weight to tensorRT engin.


# NX部署检测模型

## 步骤

### 1、安装依赖

[install_protobuf-3.8.0.sh](https://github.com/jkjung-avt/jetson_nano/blob/master/install_protobuf-3.8.0.sh)

```bash
bash install_protobuf-3.8.0.sh
```

Install pycuda

```bash
pip install pycuda==2019.1.1
```

Install onnx==1.4.1(确保是这个版本，不然会出错)

```bash
sudo pip3 install onnx==1.4.1
```

### 2、生成依赖

```bash
cd ${HOME}/project/tensorrt_demos/plugins
make
```

### 3、生成tensorRT，加速推理

```
cd ${HOME}/project/tensorrt_demos/yolo
bash darknet2onnx.sh
bash onnx2trt.sh
```

### 4、Run Yolo

```bash
python3 trt_yolo.py
```

Run on video and now we can switch between kalman filter mode to smooth the results.
```
python3 trt_yolo_with_screen.py --video /home/zq/Videos/20201022.flv -m  yolov3-416
```
