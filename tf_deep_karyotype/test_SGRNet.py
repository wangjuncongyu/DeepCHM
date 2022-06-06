# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:13:04 2022

@author: wjcongyu
"""


import os
import cv2
import time
import argparse
import numpy as np
import os.path as osp
import pandas as pd
from wrappers.chm_detector import ChmDetector
from datasets.my_dataset import MyDataset
from utils import helpers
import imageio

from cfgs.cfgs import cfg

def get_args():
    parser = argparse.ArgumentParser(description='Train the SGRANet for rotated chromosome detection')
    parser.add_argument('--use_pti', '-use_pti', type=int, default=1, help='Use prior template information for anchor cls and reg') 
    parser.add_argument('--im_root', '-im_root', type=str, default=r'D:\data\chromosome\chromosome_rotdet\test', help='the path of images to predict')
    parser.add_argument('--gtbbox_f', '-gtbbox_f', type=str, default=r'D:\data\chromosome\chromosome_rotdet\test_rbboxes.txt', help='the bboxfile of gt-bboxes')
    parser.add_argument('--load', '-load', type=str, default='/checkpoints/sgranet_dtheta15/weights_epoch120.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_root', '-save_root', type=str, default='test_results', help='the path to save detection results')
    parser.add_argument('--save_im', '-save_im', type=int, default=0, help='Save the image with pred-bboxes or not')
    parser.add_argument('--dtheta', '-dtheta', type=int, default=15, help='Theta interval for setting rotated anchorss')
    return parser.parse_args()
    
if __name__ =='__main__':
    args = get_args()
    chm_detector = ChmDetector(cfg)
    chm_detector.initialize(checkpoint_file =args.load, use_pti=args.use_pti==1, dtheta=args.dtheta)
    print(chm_detector.last_error)
    
    work_dir = 'det_rst'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
        
    save_root = osp.join(work_dir, args.save_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
        
    vis_save_path = osp.join(save_root, 'vis')
    if not os.path.exists(vis_save_path):
        os.mkdir(vis_save_path)
        
    image_root = args.im_root
    bbox_file = args.gtbbox_f
    
    #transforms = tsf.TransformCompose([tsf.Equalize_adapthist()])
    test_dataset = MyDataset(image_root, bbox_file)
    done = 0
    total = len(test_dataset)
    pred_bboxes = []
    gt_bboxes = []
    for idx in range(len(test_dataset)):
        im_file = test_dataset.get_filename(idx)
       
        print('--------------------------')
        print(done, total, im_file)
        done += 1
        
        image_org, _, i_gt_bboxs = test_dataset[idx]
        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        t = time.time()
       
        i_pred_bboxs = chm_detector.detect(image)            
        
        print('total used time:', time.time()-t)
        im_files = np.array([osp.basename(im_file) for k in range(i_pred_bboxs.shape[0])]).reshape((-1, 1))
        pred_bboxes.append(np.hstack((im_files, i_pred_bboxs)))
        
        im_files = np.array([osp.basename(im_file) for k in range(i_gt_bboxs.shape[0])]).reshape((-1, 1))
        gt_bboxes.append(np.hstack((im_files, i_gt_bboxs)))
        
        if args.save_im:
            vis_im = np.uint8(helpers.draw_bboxes2image(i_pred_bboxs, image_org, with_theta=True, txt_index=5)) 
            imageio.imsave(os.path.join(vis_save_path, os.path.basename(im_file)), np.uint8(vis_im))        
        
    #saving results to csv file
    pred_bboxes = np.concatenate(pred_bboxes, axis=0)
    
    header = ['filename', 'x', 'y','w', 'h', 'theta', 'score']
    save_data = pd.DataFrame(pred_bboxes, columns=header)
    save_data.to_csv(osp.join(save_root,'pred_bboxes.csv'),header=True, index=False)
    
    gt_bboxes = np.concatenate(gt_bboxes, axis=0)
    header = ['filename', 'x', 'y', 'w', 'h', 'theta','k','inter','label']
    #gt_bboxes = gt_bboxes[:, [0, 1, 2, 3, 4, 5,6,7, 8]]
   
    save_data = pd.DataFrame(gt_bboxes, columns=header)
    save_data.to_csv(osp.join(save_root,'gt_bboxes.csv'),header=True, index=False)
    
