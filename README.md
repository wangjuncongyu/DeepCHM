# Code, dataset, and a detailed version of the paper "Chromosome Detection in Metaphase Cell Images Using Morphological Priors". (Accepted by IEEE JBHI)
A tensorflow (>2.0) project for chromosome detection in metaphase cell images 

> This is a one-stage detector for chromosome detection using skelecton-guided rotated anchors。

### my enviroment
- Winows 10
- Anaconda python 3.7.3
- Tensorflow 2.8.0 with gpu
- cuda 11.6
- pytorch 1.12.0.dev20220504+cu116 (required for building the rotation libs, see path: tf_deep_karyotype/utils/rotation)

## dataset
Data available at the baidu cloud:https://pan.baidu.com/s/1jxAbkKYKtGg-WKcceR9w0Q
download code(提取码)：**swcf** 

## building rotation libs
[see the txt file at:  tf_deep_karyotype/utils/how to build rotated_nms.txt ]

## demo
``` bash
(1)download checkpoint file from https://pan.baidu.com/s/1BWq8TP6y7ppqlHh4tqgFhQ      (download code: zm38)
(2)put the whole checkpoints dirctor to the tf_deep_karyotype
(3)open a cmd
(4)cd tf_deep_karyotype
(5) python demo.py
```
## training
``` bash
(1)download dataset from https://pan.baidu.com/s/1jxAbkKYKtGg-WKcceR9w0Q      (download code: swcf)
(2)put the dataset to your directory. 
(3)change the source path: data_root = r'D:\data\chromosome\labelme_cis_2022' to your dataset path('tf_deep_karyotype/scripts/labelme2mytraindata_converter.py')
(4)change the save path: save_root = r'D:\data\chromosome\chromosome_rotdet_v4 to your path ('tf_deep_karyotype/scripts/labelme2mytraindata_converter.py')
(5)run tf_deep_karyotype/scripts/labelme2mytraindata_converter.py to prepare data for training 
(6)run tf_deep_karyotype/run_train.bat to train the model
```

