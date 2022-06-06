# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:01:07 2022

@author: wjcongyu
"""

import glob
import cv2
import os.path as osp
import numpy as np
from models.SGRNet import SGRNet
from utils import rbbox as box_manager
from models.backbones import BackboneFactory

class ChmDetector(object):
    def __init__(self, cfg):
        super(ChmDetector, self).__init__()     
        self.cfg = cfg
        self.model = None
        self.last_error = ''
        
    def initialize(self, checkpoint_file='last', use_pti=True, dtheta=15):
        '''
        call this function before detection on images. build network and load weights
        Input:
            checkpoint_file: the weight file to load
            use_pti: using prior template information or not in the SRGNet
            dtheta: the angle interval for setting rotated anchors
        Output:
            True: initialize done, the detector ready
            False: detector initialize failed
        '''
        try:   
            self.is_ready = False
            print('Start initiazlizing ChromosomeDetector...')
            self.cfg.ANCHOR_DETA_THETA = dtheta
            self.n_anchors = len(self.cfg.ANCHOR_RATIOS)*len(self.cfg.ANCHOR_SCALES) * (180//self.cfg.ANCHOR_DETA_THETA)
          
            self.model = SGRNet(BackboneFactory.get_backbone(self.cfg.BACKBONE),\
                                n_classes=self.cfg.NCLASSES, \
                                n_anchors = self.n_anchors,\
                                use_pti = use_pti)
            self.model.build(input_shape = tuple([self.cfg.BATCH_SIZE]+self.cfg.INPUT_SHAPE + [1]))
            
            #loading checkpoint file specified in the config file. if no checkpoint file found, try to load the latest one.
            
            if not osp.exists(checkpoint_file):
                print('No checkpoint file found:', checkpoint_file)
                
                search_roots = [osp.dirname(checkpoint_file), self.cfg.CHECKPOINTS_ROOT]
                for search_root in search_roots:
                    print('Finding lasted checkpoint file at ', search_root)
                    weights_files = glob.glob(osp.join(search_root, '*.h5'))
                    if len(weights_files) == 0:
                        print('No checkpoint file (h5 format) found!')
                        checkpoint_file = None
                        continue
                    
                    checkpoint_file = sorted(weights_files, key=lambda x: osp.getmtime(x))[-1]
                    print('Found last checkpoint file:', checkpoint_file)
                    break
        
            if checkpoint_file is None:
                return False
            
            print('Loading checkpoints from ', checkpoint_file)
            self.model.load_weights(checkpoint_file)
            
            self.is_ready = True
            print('ChromosomeDetector initializing done !')
            return True                  
                  
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)
            self.is_ready = False
            return False  
        
    def detect(self, image):
        '''
        Input:
            imgs: numpy array of size [N, H, W, C], N is the batch size, C is the channels
            
        Output:
            bboxes: a list with N elements, where each element is a 2D numpy matrix of bboxes                    
                    [[x1, y1, w1, h1, theta1, score1]
                     [x2, y2, w2, h2, theta2, score2]
                     ...
                     [xM, yM, wM, hM, thetaM, scoreM]]                                                                   
                    where M indicates the number of bbox for a sample image
        '''
                
        assert len(image.shape) == 3 or len(image.shape)==2, 'Image shape must be [H, W] or [H, W, C]'
        assert self.is_ready == True, 'SGRNet is not ready, please call <initialize> func fisrt!'
        try:
            #get the preprocessed image for feeding the model: [1, 256, 256, 3]
            
            pred_bboxes = self.get_pred_bboxes(image)
            if not (pred_bboxes is None):
            
                keep = box_manager.rotated_nms(pred_bboxes.astype('float32'),\
                                       iou_thres = self.cfg.NMS_IOU_THRES,\
                                       use_gpu=self.cfg.GPU_NMS)
            
                pred_bboxes = pred_bboxes[keep]
          
            
            return pred_bboxes
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)        
            print('!!!!!!!!!!', self.last_error)
            return None 

        
    def get_pred_bboxes(self, image):
        assert len(image.shape) == 3 or len(image.shape)==2, 'Image shape must be [H, W] or [H, W, C]'
        assert self.is_ready == True, 'SGRNet is not ready, please call <initialize> func fisrt!'
        try:
            #get the preprocessed image for feeding the model: [1, 256, 256, 3]
            oH, oW = image.shape[0:2]
            feed_im = self.__cvt2_feed_format(image.copy())
            nH, nW = feed_im.shape[1:3]            
           
            salient_pred, anchor_cls_pred, anchor_reg_pred = self.model(feed_im, False)           
           
            pred_bboxes = self.__get_pred_rbboxes(salient_pred, anchor_cls_pred, anchor_reg_pred)
          
            pred_bboxes = self.__cvtbbox2orgspace(pred_bboxes, oH, oW, nH, nW)
            return pred_bboxes
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)        
            print('!!!!!!!!!!', self.last_error)
            return None 
        
    def get_pred_maps(self, image):
        assert len(image.shape) == 3 or len(image.shape)==2, 'Image shape must be [H, W] or [H, W, C]'
        assert self.is_ready == True, 'SGRNet is not ready, please call <initialize> func fisrt!'
        try:
            #get the preprocessed image for feeding the model: [1, 256, 256, 3]
            oH, oW = image.shape[0:2]
            feed_im = self.__cvt2_feed_format(image.copy())
            nH, nW = feed_im.shape[1:3]            
           
            salient_pred, anchor_cls_pred, anchor_reg_pred = self.model(feed_im, False)           
           
            return  salient_pred, anchor_cls_pred, anchor_reg_pred
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)        
            print('!!!!!!!!!!', self.last_error)
            return None, None, None
        
    def __cvt2_feed_format(self, image):
        height, width = image.shape[0:2]
        if height != self.cfg.INPUT_SHAPE[0] or width != self.cfg.INPUT_SHAPE[1]:
            image = cv2.resize(image, (self.cfg.INPUT_SHAPE[1], self.cfg.INPUT_SHAPE[0]))
            
        if len(image.shape)==3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.reshape((1, image.shape[0], image.shape[1], -1))
        return image.astype('float32')

    
    def __get_pred_rbboxes(self, salient_pred, anchor_cls_pred, anchor_reg_pred):
       
        salient_pred = salient_pred.numpy()[0,...]
        anchor_cls_pred = anchor_cls_pred.numpy()
        anchor_reg_pred = anchor_reg_pred.numpy()
        
        skel_map = np.max(salient_pred[:, :, 0:2], axis=-1)
        anchor_locs = np.where(skel_map>self.cfg.LOCATION_THRES)
        anchor_locs = np.hstack([anchor_locs[1][:, np.newaxis], anchor_locs[0][:, np.newaxis]])
        
            
        anchors = box_manager.generate_ranchors(anchor_locs.copy(), \
                                      self.cfg.ANCHOR_BASE_SIZE, self.cfg.ANCHOR_RATIOS, \
                                      self.cfg.ANCHOR_SCALES, self.cfg.ANCHOR_DETA_THETA, self.cfg.FEATURE_STRIDE)
        
        _, H, W, A, C = anchor_cls_pred.shape
       
        xs = anchor_locs[:, 0]; ys = anchor_locs[:, 1]
       
        #default:binary classification, modify here for multi-class detection
        keep_anchor_cls_scores = anchor_cls_pred[0, ys, xs, :, 1].reshape((-1))#[n_keep x n_anchors, n_classes]    
        
        keep_anchor_reg_detas = anchor_reg_pred[0, ys, xs, ...].reshape((-1, 5))#[n_keep x n_anchors, 5] 
        
              
        keep = np.where(keep_anchor_cls_scores > self.cfg.ANCHOR_THRES)
        
        keep_anchor_cls_scores = keep_anchor_cls_scores[keep]
        keep_anchor_reg_detas = keep_anchor_reg_detas[keep]
        keep_anchors = anchors[keep]
        
        rois = self.__inverse_bboxes(keep_anchors, keep_anchor_reg_detas)
        
        # Clip predicted boxes to image.
        # if y<0 set to 0; elif y>H set to H
        rois[:, 1] = np.clip(rois[:, 1], 0, H*self.cfg.FEATURE_STRIDE)
        # if x<0 set to 0; elif x>W set to W
        rois[:, 0] = np.clip(rois[:, 0], 0, W*self.cfg.FEATURE_STRIDE)
        
        rois = np.hstack([rois, keep_anchor_cls_scores[:, np.newaxis]])
        
        
        return rois
    
   
    def __inverse_bboxes(self, anchors, anchor_detas): 
        dx, dy, dw, dh, dtheta = np.split(anchor_detas, 5, axis=1)
        
        ax, ay, aw, ah, atheta = np.split(anchors, 5, axis=1)

        px = dx*aw+ax; py = dy*ah+ay; pw = np.exp(dw)*aw; ph = np.exp(dh)*ah
        ptheta = dtheta*180.0/np.pi + atheta
      
        pbboxes = np.squeeze(np.stack((px, py, pw, ph, ptheta))).transpose(1,0)
        return pbboxes
    
    def __cvtbbox2orgspace(self, pred_bboxes, oH, oW, nH, nW):
        dx = oW/nW
        dy = oH/nH
        pred_bboxes[:, 0] *= dx
        pred_bboxes[:, 1] *= dy
        pred_bboxes[:, 2] *= dx
        pred_bboxes[:, 3] *=dy
        return pred_bboxes
    
    