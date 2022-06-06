# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:41:32 2021

@author: wjcongyu
"""
import cv2
import json
import PIL.Image
import numpy as np
from utils import helpers
from labelme import utils as labelme_utils
from transforms import TransformCompose, Resize
from skimage import morphology, feature, measure
import time
class LabelmeDataset(object):
    def __init__(self, data_root, cfg, transforms=TransformCompose([Resize(512, 512)])):
        self.cfg = cfg
        self.transforms = transforms
        self.categories = [0, 1]
        self.json_files = helpers.find_files(data_root, '.json')
        print('Found labelme files in json format:', len(self.json_files))
        
        
    def __len__(self):
        return len(self.json_files)
    
        
    def get_ncategores(self):
        return len(self.categories)
    
    
    def __getitem__(self, index):
        json_file = self.json_files[index]
      
        image, segmentations, labels = self.parse_json_file(json_file)
        
        if self.transforms:          
          
            image, segmentations, _ = self.transforms(image, segmentations)
            #image = image.numpy(); segmentations = segmentations.numpy()
           
       
        skeletons, rbboxs = self.get_skeletons_and_rbboxes(segmentations, labels)
      
        rbboxs = self.cvt_to_general_rbboxs(rbboxs)
        return image, skeletons, rbboxs
    
    def parse_json_file(self, json_file):
        image = None
        segmentations = []
        labels = []
        
        with open(json_file,'r', encoding='UTF-8') as fp:
            json_data = json.load(fp) 
            image = labelme_utils.img_b64_to_arr(json_data['imageData'])
            height, width = image.shape[:2]
            
            for shapes in json_data['shapes']:
                points=shapes['points']
                try:
                    label = int(shapes['label'])
                except:
                    label = 1
                
                mask = self.polygons_to_mask([height,width], points)
                
                labels.append(label)
                segmentations.append(mask) 
            segmentations = np.array(segmentations).transpose((1,2,0))
        return image, segmentations, labels
     
    def get_jsonfilename(self, index):
        return self.json_files[index]
        
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)
        return mask
    
    def get_skeletons_and_rbboxes(self, segmentations, labels):
        '''
        return rbbox:x, y, w, h, theta, curvature of skeleton, label
        '''
        
        skeletons = []
        edges = []
        rbboxs = []
        for i in range(segmentations.shape[-1]):
            mask = segmentations[:,:, i]
            #building bbox:x y w h angle curvature and category label
            rbbox = self.get_rotatedbox_frm_mask(mask)
            rbbox.append(labels[i])  
            curvature = self.calculate_curvature(mask)
            rbbox.append(curvature)                      
            rbboxs.append(np.array(rbbox))
            #edges and skeletons is calcualted based the feature stride
            scale = 1.0/self.cfg.FEATURE_STRIDE
            if scale!=1:
                mask = cv2.resize(mask, (0,0), fx=scale, fy=scale)
            
            skeletons.append(morphology.skeletonize(mask.copy()))
            edges.append(feature.canny(mask.copy(), sigma=1.0))
            
        rbboxs = np.array(rbboxs)
        N_bboxs = rbboxs.shape[0]
        
        insection_labels = np.zeros((N_bboxs))
       
        for i in range(N_bboxs):
            if insection_labels[i] == 1:
                continue
            skeleton1 = np.uint8(skeletons[i].copy())
            for j in range(i+1, N_bboxs):                
                skeleton2 = np.uint8(skeletons[j].copy())
                
                if np.max(skeleton1*skeleton2)==1:
                    insection_labels[i] = 1
                    insection_labels[j] = 1
                    
        
        rbboxs = np.hstack([rbboxs, np.expand_dims(insection_labels, axis=-1)])
                
            
        salient_mask = np.sum(np.array(skeletons), axis=0)#label 1 for skeleton points
        salient_mask[salient_mask>1] = 2#label 2 for intersection points
        cnt_xs = np.int16(rbboxs[:, 0]//self.cfg.FEATURE_STRIDE); cnt_ys = np.int16(rbboxs[:, 1]//self.cfg.FEATURE_STRIDE)
        salient_mask[cnt_ys, cnt_xs] =2 #label 2 for bbox center points
        
        edge_mask = np.sum(np.array(edges), axis=0)
        edge_mask[edge_mask>1] = 1        
        salient_mask[edge_mask==1] = 3#label 2 for edge points
       
        return np.uint8(salient_mask), rbboxs
    
    def get_skeletons_keypoints_rbboxes(self, segmentations, labels):
        '''
        return rbbox:x, y, w, h, theta, curvature of skeleton, label
        '''
        dilate_masks = []
        skeletons = []
        rbboxs = []
        for i in range(segmentations.shape[-1]):
            mask = segmentations[:,:, i]
            rbbox = self.get_rotatedbox_frm_mask(mask)
            curvature = self.calculate_curvature(mask)
            rbbox.append(curvature)
            rbbox.append(labels[i])            
            rbboxs.append(np.array(rbbox))
            #tempalte size is determined by the feature stride
            scale = 1.0/self.cfg.FEATURE_STRIDE
            if scale!=1:
                mask = cv2.resize(mask, (0,0), fx=scale, fy=scale)
            
            disk_size= 2
            dilate_masks.append(morphology.binary_dilation(mask.copy(), morphology.disk(disk_size)))
          
            skeletons.append(morphology.skeletonize(mask.copy()))
            
        skeleton_mask = np.sum(np.array(skeletons), axis=0)#label 1 for skeleton points
        skeleton_mask[skeleton_mask>1] = 3#label for intersection points
        
        overlap_mask = np.sum(np.array(dilate_masks), axis=0)
        overlap_mask[overlap_mask<=1] = 0
        overlap_mask[overlap_mask>1] = 1
        region_labels = measure.label(overlap_mask, connectivity=2)	
        region_props = measure.regionprops(region_labels)
        
        for prop in region_props:
            y1, x1, y2, x2 = prop.bbox
            skeleton_patch = skeleton_mask[y1:y2, x1:x2]
            if np.max(skeleton_patch)==3 or prop.area<4:
                overlap_mask[region_labels==prop.label] = 0
                
        connect_mask = morphology.skeletonize(overlap_mask)
        skeleton_mask[connect_mask>0] = 2#label for connecting regions
    
        return np.uint8(skeleton_mask), np.array(rbboxs)
                
 
    def get_rotatedbox_frm_mask(self, mask):
        (contours, _) = \
        cv2.findContours((255*mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) == 0:
                continue
        rect = cv2.minAreaRect(contours[0])
        rbbox =[rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]
        
        return rbbox
    
    def calculate_curvature(self, mask):
        skeleton = morphology.skeletonize(mask.copy())
        pts = np.where(skeleton>0)
        ys = pts[0]
        xs = pts[1]
        if ys.shape[0]<6:
            return 0.1
        k, b = np.polyfit(xs, ys, 1)
        distance = np.abs(k*xs-ys+b)/np.sqrt(k*k+1)
        return np.mean(distance)
            
    
    def cvt_to_general_rbboxs(self, rbboxs):
        '''
        convert opencv rbbox in range [-90, 0) to general rbbox in range [0, 180)
        '''
        for idx, rbbox in enumerate(rbboxs):
            x, y, w, h, theta = rbbox[0:5]
       
            if w < h:
                rbboxs[idx, 2] = h
                rbboxs[idx, 3] = w
                rbboxs[idx, 4] += 90
            else:
                rbboxs[idx, 4] += 180
        
        return rbboxs
        
        
            
            
    
    