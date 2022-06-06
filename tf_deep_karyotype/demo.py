# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:11:28 2022

@author: wjcongyu
"""


import cv2
import imageio
import numpy as np
from utils import helpers
import matplotlib.pyplot as plt
from wrappers.chm_detector import ChmDetector



from cfgs.cfgs import cfg

chm_detector = ChmDetector(cfg)

load = 'checkpoints/sgranet_dtheta15/weights_epoch10.h5'
chm_detector.initialize(checkpoint_file =load, use_pti=True, dtheta=cfg.ANCHOR_DETA_THETA)

im_org = imageio.imread('demo_images/70_2_737_279_0.683.png')
print('image:', im_org.shape)

im_gray = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)

pred_bboxs = chm_detector.detect(im_gray)     

#filter out pred-bboxes with small scores
T = 0.5
keep = np.where(pred_bboxs[:, 5]>=T)
pred_bboxs = pred_bboxs[keep]

vis_im = np.uint8(helpers.draw_bboxes2image(pred_bboxs, im_org, with_theta=True, txt_index=5)) 

plt.imshow(vis_im)
plt.axis('on')  
plt.title('Chromosome detection')  
 
plt.show()


