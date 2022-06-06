# -*- coding: utf-8 -*-
'''
@Date          : 2022-04-13 21:41:33
@Author        : wjcongyu
@Contact       : wangjun@zucc.edu.cn
@Copyright     : 2022 ZUCC
@License       : CC BY-NC-SA 4.0
@Last Modified : wjcongyu 2022-04-13 21:41:34
@Des           : None

@Log           : None

'''
import tensorflow as tf
from tensorflow.keras import layers as KL

class ASPP(tf.keras.Model):
    def __init__(self, nfilter):
        super(ASPP, self).__init__()
        self.dilate_x1 = KL.Conv2D(nfilter//4, (1, 1), dilation_rate = 1, activation = 'relu', padding='same')
        self.dilate_x2 = KL.Conv2D(nfilter//4, (3, 3), dilation_rate = 6, activation = 'relu', padding='same')
        self.dilate_x3 = KL.Conv2D(nfilter//4, (3, 3), dilation_rate = 12, activation = 'relu', padding='same')
        self.dilate_x4 = KL.Conv2D(nfilter//4, (3, 3), dilation_rate = 18, activation = 'relu', padding='same')
        
        self.feature_levelx = KL.Conv2D(nfilter//4, (2, 2), dilation_rate = 1, activation = 'relu', padding='same')
        self.smooth_x = KL.Conv2D(nfilter, (1, 1), dilation_rate = 1, activation = 'relu', padding='same')
    
    def call(self, x, training):
        dx1 = self.dilate_x1(x)
        dx2 = self.dilate_x2(x)
        dx3 = self.dilate_x3(x)
        dx4 = self.dilate_x4(x)
        agvx = tf.reduce_mean(x, [1, 2], keepdims=True)
        agvx = self.feature_levelx(agvx)
        agvx = tf.image.resize(images = agvx, size=[tf.shape(x)[1], tf.shape(x)[2]])
        di_concat = KL.Concatenate(axis=-1)([dx1, dx2, dx3, dx4, agvx])
        return self.smooth_x(di_concat)
    
#the subnet head for box classification and regress
class SubNetHead(tf.keras.Model):
    def __init__(self, n_classes, n_anchors):
        super(SubNetHead, self).__init__() 
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.x1 = KL.Conv2D(256, (1, 1), activation = 'relu', padding='same')  
        self.x2 = KL.Conv2D(256, (3, 3), activation = 'relu', padding='same')  
        #self.alpha = KL.Conv2D(128, (1, 1), activation=None, kernel_initializer='zeros', use_bias=False)
        #extractng multiscale features using ASPP
        self.ASPP_x = ASPP(256)
        self.logit = KL.Conv2D(n_anchors*self.n_classes, (3, 3), padding='same', \
                               kernel_initializer='zeros', activation=None, use_bias=False)
        
       
    def call(self, x, prior_map=None, training=True):        
        if not (prior_map is None):
            x = KL.Concatenate(axis=-1)([x, self.x1(prior_map)])
        #x1 = self.x1(x)
        x2 = self.x2(x)
        
        aspp_x = self.ASPP_x(x2, training)
        logit = self.logit(aspp_x)
      
        return logit

        
class SGRNet(tf.keras.Model):
    def __init__(self, backbone, n_classes, n_anchors, use_pti=True):
        super(SGRNet, self).__init__()     
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        
        self.n_reg_params = 5
        self.use_pti = use_pti
        self.backbone = backbone
       
        self.x = KL.Conv2D(256, (3, 3), activation = 'relu', padding='same')        
                
        #two channels: channel0 for skeletons, channel1 for intersecton points and channel 3 for edges
        self.salient_map = KL.Conv2D(3, (3, 3),kernel_initializer='zeros', padding='same', activation='sigmoid', use_bias=False)
        
        self.cls_subnet = SubNetHead(self.n_classes, self.n_anchors)
        self.reg_subnet = SubNetHead(self.n_reg_params, self.n_anchors)
        
    def call(self, ims, training=True):
        features = self.backbone(ims, training)
        
        x = self.x(features)
        salient_map = self.salient_map(x)        
       
        pti = salient_map if self.use_pti else tf.zeros_like(salient_map)
       
        cls_logit = self.cls_subnet(x, pti)    
       
        reg_logit = self.reg_subnet(x, pti)
        _, H, W, _ = cls_logit.shape
        
        cls_logit = tf.reshape(cls_logit, [-1, H, W, self.n_anchors, self.n_classes])       
        cls_score = KL.Softmax()(cls_logit)
        
        reg_logit = tf.reshape(reg_logit, [-1, H, W, self.n_anchors, self.n_reg_params])
        
        return salient_map, cls_score, reg_logit
    
    def print_summary(self):
        print('-------------- Network achitecture --------------') 
        print(self.backbone.summary()); print(self.cls_subnet.summary());
        print(self.reg_subnet.summary()); print(self.summary()) 


    

   