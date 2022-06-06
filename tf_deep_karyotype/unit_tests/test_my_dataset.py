# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:40:28 2021

@author: wjcongyu
"""

import _init_pathes
import os
import numpy as np
import os.path as osp
import imageio
from cfgs.cfgs import cfg
from datasets.my_dataset import MyDataset
from utils.helpers import draw_bboxes2image
import datasets.transforms as tsf 

image_root = r'D:\data\chromosome\chromosome_rotdet\test'
annotation_file = r'D:\data\chromosome\chromosome_rotdet\test_rbboxes.txt'
transforms = tsf.TransformCompose([ tsf.RandomFlip(),\
                                    tsf.RandomRotate(),\
                                    tsf.RandomBrightness(),\
                                    tsf.RandomContrast(),\
                                    tsf.RandomGamma(),\
                                    tsf.RandomSaturation()])
dataset = MyDataset(cfg.TRAIN_DATA_ROOT, cfg.TRAIN_BBOX_FILE, transforms)
save_path = 'test_my_dataset'
if not osp.exists(save_path):
    os.mkdir(save_path)
    
for idx in range(len(dataset)):
    image, target, rbbox = dataset[idx]
    image = np.uint8(image)
    im_file = dataset.get_filename(idx)
    
    '''imageio.imsave(osp.join(save_path, osp.basename(im_file)),image)'''
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.png', '_skel.png')), np.uint8(target[:,:,0]*255))
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.png', '_edge.png')), np.uint8(target[:,:,2]*255))
    image_box = draw_bboxes2image(rbbox, image, True)
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.png', '_bbox.png')), image_box)