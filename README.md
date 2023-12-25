# TODD - Thermal Object Detection for autonomous Driving
This repository contains code to experiment with the response time of an Nvidia Jetson Nano 2GB in object detection. Specifically, a model trained to recognize objects from thermal images was used in a separate processor with a more powerful GPU and, once the model was obtained, imported, optimized with NVIDIA TensorRT and run on the Jetson Nano.

There are different phases and thus, also different libraries and frameworks. Part of the study was also to identify the smoothest workflow in the current state. In particular, we undertook two development strides: 
- Tensorflow, in which case you stay in the same environment in all phases:
    - Importing the pre-trained SSD Mobilenet V2 network into Tensorflow Hub.
    - Model training in Tensorflow
    - Exporting in SavedModel format 
    - Importing into Jeston again with Tensorflow 
    - Optimization with TF-TensorRT
    - Inference in Tensorflow
- Pytorch, here, however, you have to go through the ONNX format
    - Importing the pre-trained SSDLite Mobilenet V3 network into Pytorch Hub
    - Training the model with Pytorch
    - Exporting to ONNX format
    - Optimization in TensorRT
    - Inference in TensorRT

## Results
The results were satisfactory for the flow in Tensorflow. The response times of the Jetson Nano were around 10 FPS. Considering that it was tested on a 60€ board and that, only three years ago a similar experiment was undertaken with a RaspberryPi3 ([Real-Time Human Detection as an Edge Service Enabled by a Lightweight CNN](https://ieeexplore.ieee.org/document/8473387) ). In that case, the best result was 1.82 FPS. This result could be further improved by exploiting the C++ API of Tensorflow or, again, by being able to optimize the whole model in TensorRT.

Regarding accuracy, the COCO metric was usat. The table with the results between the two architectures compared is given here
|Metric|IoU|Area|maxDets|SSDLite Mobilenet V3|SSD Mobilenet V2|
|------|---|----|-------|-----|------|
|Average Precision  (AP)|0.50:0.95|all|100|0.076|0.111|
|Average Precision  (AP)|0.50|all|100|0.173|0.3|
|Average Precision  (AP)|0.75|all|100|0.061|0.07|
|Average Precision  (AP)|0.50:0.95|small|100|0.007|0.045|
|Average Precision  (AP)|0.50:0.95|medium|100|0.095|0.2497|
|Average Precision  (AP)|0.50:0.95|large|100|0.486|0.4244|
|Average Recall     (AR)|0.50:0.95 |all|1|0.054|0.089|
|Average Recall     (AR)|0.50:0.95 |all|10|0.147|0.22|
|Average Recall     (AR)|0.50:0.95 |all|100|0.205|0.23|
|Average Recall     (AR)|0.50:0.95 |small|100|0.099|0.126|
|Average Recall     (AR)|0.50:0.95 |medium|100|0.257|0.4452|
|Average Recall     (AR)|0.50:0.95 |large|100|0.640|0.6393|

A test performed on the Jetson Nano is shown below. The frame rate is reduced to about 3 FPS because the result processing and bounding box drawing is all done on the board, and this takes about 300 ms.

![video sample](doc/sample-output.gif)

Unfortunately, at the time of writing this article, development through Pytorch could not be completed. The obstacle was due to the conversion through the ONNX format, in fact, TensorRT still does not support all modules in the original network that are encoded in that format. To continue, one would have had to manually intervene in the ONNX file before providing it to TensorRT, but the process would have been time-consuming and complex. The same did not happen in Tensorflow, because TF-TRT takes care of converting only the compatibli modules leaving the others unchanged. Given the continued input from NVIDIA and the continuous updates, we expect the conversion to be increasingly smooth in the future.
