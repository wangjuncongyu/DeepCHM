# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:53:34 2021

@author: wjcongyu
"""

import _init_pathes
import os
import imageio
import numpy as np
import os.path as osp

from datasets import transforms as tsf 
from datasets.labelme_dataset import LabelmeDataset
from utils import helpers
from cfgs.cfgs import cfg

#the source data root of the labeme annotations
data_root = r'D:\data\chromosome\CISeg2022'

exp_height, exp_width = cfg.INPUT_SHAPE
transforms = tsf.TransformCompose([tsf.Resize(exp_height, exp_width)])

#the path to save the our dataset
save_root = r'D:\data\chromosome\chromosome_rotdet2'
if not osp.exists(save_root):
    os.mkdir(save_root)
    
for subset in ['train', 'test']:
    
    save_path = osp.join(save_root, subset)
    if not osp.exists(save_path):
        os.mkdir(save_path)
        
    subset_data_root = osp.join(data_root, subset)
    dataset = LabelmeDataset(subset_data_root, cfg, transforms)
    bbox_file = open(os.path.join(save_root, subset+'_rbboxes.txt'), 'w', encoding='utf-8')
    N = len(dataset)
   
    for n in range(N):
        print(n, N, subset)
        json_file = dataset.get_jsonfilename(n)
       
        
        image, skeletons, rbboxes = dataset[n]
        gaus = helpers.generate_gaussian_targets(skeletons)
        
        patient = osp.basename(osp.dirname(json_file))
        
        
        im_file = patient+'_'+osp.basename(json_file).replace('.json', '.png')
        imageio.imsave(osp.join(save_path, im_file), np.uint8(image))
      
        target_file = im_file.replace('.png', '_target.npz')
        np.savez(osp.join(save_path, target_file), gaus)
        for box in rbboxes:
            x, y, w, h, angle, label, k, inter = box
            #assert w>1 and h>1, 'w and h ==0'
            if w<3 or h<3:
                continue
            write_line = '{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(im_file, \
                          round(x,1), round(y,1), round(w,1), round(h,1), round(angle,1), int(label), round(k, 2), int(inter))
    
            bbox_file.write(write_line)
        
    bbox_file.close()


