# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:45:53 2021

@author: wjcongyu
"""
from easydict import EasyDict as edict

cfg = edict()
cfg.TRAIN_DATA_ROOT = r'D:\data\chromosome\chromosome_rotdet\train'
cfg.TRAIN_BBOX_FILE = r'D:\data\chromosome\chromosome_rotdet\train_rbboxes.txt'
cfg.CHECKPOINTS_ROOT = 'checkpoints'

#training phase
cfg.INPUT_SHAPE = [512,768]
cfg.BACKBONE = 'ResNet_101'
cfg.NCLASSES = 2 #background + foreground
cfg.LOCATION_THRES = 0.2

cfg.ANCHOR_BASE_SIZE = 5.0
cfg.ANCHOR_RATIOS = [2.0, 4.0]#the ratio must >=1.0
cfg.ANCHOR_SCALES = [2.0, 5.0]
cfg.ANCHOR_DETA_THETA = 15
cfg.FEATURE_STRIDE = 4
cfg.ANCHOR_POS_IOU_THRES = 0.6 #positive anchors with large IoU to any gt-box
cfg.ANCHOR_NEG_IOU_THRES = 0.4 #negative anchors with small IoU to any gt-box
cfg.ANCHOR_MAX_NUM_POSITIVES = 1024

#cfg.ANCHOR_NUM_NEGATIVES = 300
cfg.LR = 0.0001
cfg.DECAY_STEPS = 518
cfg.DECAY_RATE = 0.95
cfg.EPOCHS = 120
cfg.STEPS_PER_EPOCH = 518
cfg.BATCH_SIZE =1
cfg.MAX_KEEPS_CHECKPOINTS = 1

#testing phase
#cfg.CHECKPOINT_FILE = 'checkpoints/SGRA/weights_epoch100.h5'
#cfg.CENTER_THRES = 0.05
cfg.ANCHOR_THRES = 0.1
cfg.NMS_IOU_THRES = 0.2
cfg.GPU_NMS = True #change to False if GPU not accessed