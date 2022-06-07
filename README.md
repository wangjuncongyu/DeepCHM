# DeepCHM
A tensorflow (>2.0) project for chromosome detection in metaphase cell images 

> This is a one-stage detector for chromosome detection using skelecton-guided rotated anchors。

### my enviroment
- Winows 10
- Anaconda python 3.7.3
- Tensorflow 2.8.0 with gpu
- cuda 11.6
- pytorch 1.12.0.dev20220504+cu116 (required for building the rotation libs, see path: tf_deep_karyotype/utils/rotation)

## dataset
[the dataset is coming soon !]

## building rotation libs
[see the txt file at:  tf_deep_karyotype/utils/how to build rotated_nms.txt ]

## demo
``` bash
(1)download checkpoint file from https://pan.baidu.com/s/1BWq8TP6y7ppqlHh4tqgFhQ      (dowload code: zm38)
(2)open a cmd
(3)cd tf_deep_karyotype
(4) python demo.py
```
