<!--
 * @Author: your name
 * @Date: 2020-10-23 13:19:32
 * @LastEditTime: 2020-10-23 13:22:03
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /detect_board/home/zq/zq/git-space/tensorrt_about/yolov3_tensorrt/yolo/README.md
-->
<pre><font color="#729FCF"><b>.</b></font>
├── <font color="#729FCF"><b>darknet</b></font>
│   ├── yolov3-416.cfg
│   ├── yolov3-416.onnx
│   ├── yolov3-416.trt
│   └── yolov3-416.weights
├── <font color="#729FCF"><b>darknet2onnx</b></font>
│   ├── darknet2onnx.sh
│   ├── <font color="#729FCF"><b>__pycache__</b></font>
│   │   └── plugins.cpython-36.pyc
│   └── yolo_to_onnx.py
├── <font color="#729FCF"><b>onnx2trt</b></font>
│   ├── onnx2trt.py
│   └── onnx2trt.sh
├── plugins.py
├── <font color="#729FCF"><b>__pycache__</b></font>
│   └── plugins.cpython-36.pyc
├── README.md
└── requirements.txt</pre>

This is the **tree** of the file.
* The **darknet** file contains model weights of yolov3 detection and tensorRT engine.
* The **darknet2onnx** converts model weights to onnx format.
* The **onnx2trt** build a TensorRT engine from an ONNX file.
In each file, there are bash files which indicate how to run the command.
