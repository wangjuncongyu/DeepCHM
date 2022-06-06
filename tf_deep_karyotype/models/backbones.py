# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:26:49 2021

@author: wjcongyu
"""
import tensorflow as tf
from tensorflow.keras import layers as KL
from .residual_block import make_basic_block_layer, make_bottleneck_layer
class BackboneFactory(object):
    @staticmethod
    def get_backbone(self, backbone_name='ResNet_50'):
        if backbone_name == 'ResNet_18':
            return ResNetTypeI(layer_params=[2, 2, 2, 2])
        elif backbone_name == 'ResNet_34':
            return ResNetTypeI(layer_params=[3, 4, 6, 3])
        elif backbone_name == 'ResNet_50':
            return ResNetTypeII(layer_params=[3, 4, 6, 3])
        elif backbone_name == 'ResNet_101':
            return ResNetTypeII(layer_params=[3, 4, 23, 3])
        elif backbone_name == 'ResNet_152':
            return ResNetTypeII(layer_params=[3, 8, 36, 3])
        

class ConvBnRelu(tf.keras.Model):
    def __init__(self, nfilter, kernel=(3, 3), stride=(1, 1), dilation_rate = (1, 1)):
        super(ConvBnRelu, self).__init__()
        self.x = KL.Conv2D(nfilter, kernel, stride, dilation_rate = dilation_rate, padding='same')  
        self.bn = KL.BatchNormalization(axis=-1)     
        self.ac = KL.ReLU()
        
    def call(self, x, training):
        return self.ac(self.bn(self.x(x), training))



        
class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()       

        #bottom-->up layers
        self.C1 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[0])
        self.C2 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[1],
                                             stride=2)
        self.C3 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[2],
                                             stride=2)
        self.C4 = make_basic_block_layer(filter_num=512,
                                             blocks=layer_params[3],
                                             stride=2)

        #smooth layers
      
        self.S4 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")
        
        #lateral layers
        self.L1 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L2 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L3 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L4 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        #self.L5 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
    
       
    def call(self, inputs, training=True, mask=None):
        C0 = tf.nn.relu(self.bn1(self.conv1(inputs), training = training))#stride 2
        #C0 = tf.nn.max_pool2d(C0, ksize=3, strides=2, padding='SAME')#stride 4
        #botton-->up
        C1 = self.C1(C0, training=training)
        C2 = self.C2(C1, training=training)
        C3 = self.C3(C2, training=training)
        C4 = self.C4(C3, training=training)
        
        #top-->down
        P4 = self.L1(C4)
        P3 = self._upsample_add(P4, self.L2(C3))
        P2 = self._upsample_add(P3, self.L3(C2))
        P1 = self._upsample_add(P2, self.L4(C1))
        #P0 = self._upsample_add(P1, self.L5(C0))
        #smooth
      
        P1 = self.S4(P1)
        return P1
    
    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return KL.Add()([x, y])


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.C1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.C2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.C3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.C4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        
      
        #lateral layers
        self.L1 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L2 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L3 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        self.L4 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        #self.L5 = KL.Conv2D(256, (1, 1), (1, 1),  padding="valid")
        
        #smooth layers       
        self.S3 = KL.Conv2D(256, (3, 3), (1, 1),  padding="same")
       
    def call(self, inputs, training=True, mask=None):
        C0 = tf.nn.relu(self.bn1(self.conv1(inputs), training = training))
        #C0 = tf.nn.max_pool2d(C0, ksize=3, strides=2, padding='SAME')
        #botton-->up
        C1 = self.C1(C0, training=training)
        C2 = self.C2(C1, training=training)
        C3 = self.C3(C2, training=training)
        C4 = self.C4(C3, training=training)
        
        #top-->down
        P4 = self.L1(C4)
        P3 = self._upsample_add(P4, self.L2(C3))
        P2 = self._upsample_add(P3, self.L3(C2))
        #P1 = self._upsample_add(P2, self.L4(C1))
        #P0 = self._upsample_add(P1, self.L5(C0))
        #smooth
      
        P2 = self.S3(P2)
        return P2
    
    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        x = tf.image.resize(x, size=(H, W), method='bilinear')
        return KL.Add()([x, y])
       


