# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:41:32 2021

@author: wjcongyu
"""


import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image

class MyDataset(object):
    def __init__(self, data_root, bbox_file, transforms=None):
        self.data_root = data_root
        self.bbox_file = bbox_file
        assert osp.exists(self.bbox_file), 'No bbox file found:'+self.bbox_file
        
        self.transforms = transforms
        self.categories = [0, 1]
      
        self.__load_bbox_file(self.bbox_file)
        
        
    def __len__(self):
        return len(self.file_pairs)
    
        
    def get_ncategores(self):
        return len(self.categories)
    
    def get_filename(self, index):
        return self.file_pairs[index][0]
    
    def __load_bbox_file(self, bbox_file):
        self.bboxes = {}
        self.file_pairs = []
        self.categories = [0]
        print('Start parsing bbox file... ', bbox_file)
        annos = pd.read_table(bbox_file, sep=' ',header=None)
        for i in annos.index.values:
            filename, x, y, w, h, angle, label, k, inter = annos.iloc[i, 0:9]
            if w<3 or h<3: continue
        
            img_file = osp.join(self.data_root, filename)
            target_file = img_file.replace('.png', '_target.npz')
            if (not osp.exists(img_file)) or (not osp.exists(target_file)):
                continue
            if filename not in self.bboxes:
                self.bboxes[filename] = []
                self.file_pairs.append([img_file, target_file])
                
            self.bboxes[filename].append([x, y, w, h, angle, k, inter, label])
            if label not in self.categories: self.categories.append(label)
        return self.bboxes
    
    
    def __getitem__(self, index):
        im_file, target_file = self.file_pairs[index]
        image = np.array(Image.open(im_file), np.uint8)
        target = np.load(target_file, allow_pickle=True)['arr_0']
        #skeleton_gau = np.load(skel_file, allow_pickle=True)['arr_0']
        rbboxs = np.array(self.bboxes[osp.basename(im_file)])
        if self.transforms:
            image, target, rbboxs =  self.transforms(image, target, rbboxs)
          
        return image, target, rbboxs
    
   