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

### 4、奔跑Yolo

```bash
python3 trt_yolo.py
```


