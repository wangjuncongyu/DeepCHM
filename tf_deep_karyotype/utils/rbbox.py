# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 21:36:13 2021

@author: wjcongyu
"""

import numpy as np
from .rotation.rbbox_overlaps import rbbx_overlaps
from .rotation.rotate_cython_nms import rotate_cpu_nms
from .rotation.rotate_polygon_nms import rotate_gpu_nms

def generate_dense_anchor_loc(width, height, stride=1, mask=None):   
    x = np.arange(0, width, stride)
    y = np.arange(0, height, stride)
    xs, ys = np.meshgrid(x, y, indexing='ij')
    xs = xs.reshape(-1, 1); ys = ys.reshape(-1, 1)
    if mask is not None:
        maskt = np.zeros_like(mask)
        maskt[ys, xs] = 1
        maskt = maskt*mask
        keep_locs = np.where(maskt==1)
        keep_locs = np.hstack([keep_locs[1][:, np.newaxis], keep_locs[0][:, np.newaxis]])
        return keep_locs
    else:
        return np.hstack([xs, ys])
    
def generate_ranchors(anchor_locs, base_size = 6, ratios=[2.0, 4.0], scales = [2, 4, 6], dtheta = 15, stride= 2):
    '''
    Generate the rotated anchors for each anchor location.
    '''
   
    thetas = np.array([k for k in range(0, 180, dtheta)])
    n_ratios = len(ratios); n_scales = len(scales); n_thetas = len(thetas)
    
    base_anchors = np.zeros((n_ratios*n_scales*n_thetas, 5), dtype=np.float32)
    
    for i in range(n_ratios):
        for j in range(n_scales):
            w = base_size * scales[j] * np.sqrt(ratios[i])
            h = base_size * scales[j] * np.sqrt(1. / ratios[i])
            
            for k in range(n_thetas):         
                
                index = i * n_scales * n_thetas + j * n_thetas + k
                base_anchors[index] = (0, 0, w, h, thetas[k])
                
    
    n_locs = anchor_locs.shape[0]
    n_base_anchors = base_anchors.shape[0]
    anchor_locs = np.repeat(anchor_locs, n_base_anchors, axis=0)
    
    base_anchors = np.tile(base_anchors, (n_locs,1))
    
    anchors = np.hstack([(base_anchors[:, 0:2] + anchor_locs)*stride, base_anchors[:,2:]])
    
    return anchors


def calculate_rbbox_iou(boxes1, boxes2):
    
    '''
    Calculate the IoU between boxes1 and boxes2 in center format [x, y, w, h] 
    Input:
        boxes1: numpy array of boxes 1
        boxes2: numpy array of boxes 2       
        with_theta: 'True' means box is [x, y, w, h, theta], else [x, y, w, h]
               
    Output:
        ious: the numpy array of ious between each box
    '''
   
    nboxes_per_batch = 2048
    n_batch = boxes1.shape[0] // nboxes_per_batch
    ious = []
    boxes2 = np.ascontiguousarray(boxes2, dtype=np.float32)
    for i in range(n_batch+1):
        start = i*nboxes_per_batch
        end = min(boxes1.shape[0], (i+1)*nboxes_per_batch)
        i_anchors = boxes1[start:end, ...]
        if i_anchors.shape[0]==0:
            continue
        #print('iou calc:',i_anchors.shape[0], boxes2.shape[0] )
        i_anchors = np.ascontiguousarray(i_anchors, dtype=np.float32)
        i_ious = rbbx_overlaps(i_anchors, boxes2)
        ious.append(i_ious)
    ious = np.vstack(ious).astype('float32')
    #ious[ious > 1.01] = 0.0
    return ious
    
def rotated_nms(bboxes, iou_thres = 0.1, use_gpu=True):
    '''
    Non-maximum suppression for removing overlaped rotated-bboxes with 
    center format [x, y, w, h, theta, score]
    Input: 
        bboxes: the predicted bbox
        
    '''
    
    assert bboxes.shape[1]>=6, 'no score given in the box at index [5]'  
    if use_gpu:
        return rotate_gpu_nms(bboxes, iou_thres, int(0))
    else:
        return rotate_cpu_nms(bboxes, iou_thres)