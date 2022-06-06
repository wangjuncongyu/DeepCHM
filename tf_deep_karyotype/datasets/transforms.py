# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 17:00:50 2021

@author: wjcongyu
"""

import random
import numpy as np
import tensorflow as tf
from skimage import exposure
class TransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, bbox=None):
        for t in self.transforms:
            image, target, bbox = t(image, target, bbox)
            
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        if (not target is None) and  isinstance(target, tf.Tensor):
            target = target.numpy()
        return image, target, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Equalize_adapthist(object):   
    def __init__(self, kernel_size=128, nbins=256, clip_limit =0.01):
        self.kernel_size = kernel_size
        self.nbins = nbins
        self.clip_limit = clip_limit
        
    def __call__(self, image, target=None, bbox=None):        
        image = self.enhance_contrast(image)
        return image, target, bbox 
    
    def enhance_contrast(self, image):
        image = np.asarray(image)
        image = exposure.equalize_adapthist(image,\
                                            kernel_size=self.kernel_size,\
                                            nbins=self.nbins, \
                                            clip_limit=self.clip_limit)
        return np.uint8(image*255)
    
class Resize(object):
    def __init__(self, t_height, t_width):
        self.t_height = t_height
        self.t_width = t_width
        
    def __call__(self, image, target=None, bbox=None):        
        image = tf.image.resize(image, [self.t_height, self.t_width])
        if target is not None: 
            target = tf.image.resize(target, [self.t_height, self.t_width])
        return image, target, bbox 
    
class ResizeWithPad(object):
    def __init__(self, t_height, t_width):
        self.t_height = t_height
        self.t_width = t_width
        
    def __call__(self, image, target=None, bbox=None):        
        image = tf.image.resize_with_pad(image, self.t_height, self.t_width)
        if target is not None: 
            target = tf.image.resize_with_pad(target, self.t_height, self.t_width)
        return image, target, bbox
  
class RandomFlip(object):
    def __call__(self, image, target=None, bbox=None):
        random_flip = random.randint(0, 2)
        if random_flip == 0:
            return image, target, bbox
        elif random_flip == 1:
            image = tf.image.flip_left_right(image)
            if target is not None: 
                target = tf.image.flip_left_right(target)
                
            if bbox is not None:
                bbox[:, 0] = image.shape[1] - bbox[:, 0]
                bbox[:, 4] = 180-bbox[:,4]
            return image, target, bbox
        else:
            image = tf.image.flip_up_down(image)
            if target is not None:
                target = tf.image.flip_up_down(target)
            if bbox is not None:
                bbox[:, 1] = image.shape[0] - bbox[:, 1]
                bbox[:, 4] = 180-bbox[:,4]
            return image, target, bbox
        
class RandomRotate(object):        
    def __call__(self, image, target=None, bbox=None):
        random_angle = random.randint(0, 3) * 90
        if random_angle in [0, 90, 270] :
            return image, target, bbox
           
        H, W = image.shape[0:2]
        # rotate by random_angle degrees.
        image = tf.image.rot90(image, random_angle//90)
        if target is not None: 
            target = tf.image.rot90(target, random_angle//90)
        if bbox is not None:
            x, y= bbox[:, 0].copy(), bbox[:, 1].copy()
            if random_angle == 90:
                bbox[:, 0] = H - y
                bbox[:, 1] = x
                bbox[:, 4] = 180 - bbox[:, 4]
            elif random_angle == 180:
                bbox[:, 0] = W - x
                bbox[:, 1] = H - y
            elif random_angle == 270:
                bbox[:, 0] = y
                bbox[:, 1] = W - x
                bbox[:, 4] = 180 - bbox[:, 4]
               
        return image, target, bbox
    
class RandomBrightness(object):
    def __init__(self, \
                 brightness= [0.0, 0.0, 0.0, 0.02, 0.04, 0.06, 0.08]):
        self.brightness = brightness
        
    def __call__(self, image, target=None, bbox=None):
        brightness_factor = self.brightness[random.randint(0, len(self.brightness)-1)]
      
        brightness_factor = max(0, brightness_factor)
        brightness_factor = min(1.0, brightness_factor)
      
        return tf.image.adjust_brightness(image, brightness_factor), target, bbox
    
class RandomContrast(object):
    def __init__(self, \
                 contrasts=[1.0, 1.0, 1.0, 1.2, 1.4, 1.6]):
        self.contrasts = contrasts
        
    def __call__(self, image, target=None, bbox=None):
        contrast_factor = self.contrasts[random.randint(0, len(self.contrasts)-1)]
        contrast_factor = max(1.0, contrast_factor)
        contrast_factor = min(2.0, contrast_factor)
        return tf.image.adjust_contrast(image, contrast_factor), target, bbox
    
class RandomGamma(object):
    def __init__(self, gamma=[1.0, 1.0, 1.0, 1.1, 1.2, 1.3]):
        self.gamma = gamma
        
    def __call__(self, image, target=None, bbox=None):
        gamma_factor = self.gamma[random.randint(0, len(self.gamma)-1)]
        gamma_factor = max(1.0, gamma_factor)
        gamma_factor = min(2.0, gamma_factor)
       
        return tf.image.adjust_gamma(image, gamma_factor), target, bbox
    
class RandomSaturation(object):
    def __init__(self, saturation=2.0):
        self.saturation = saturation
        
    def __call__(self, image, target=None, bbox=None):
        saturation_factor = random.normalvariate(1, self.saturation)
        saturation_factor = max(0, saturation_factor)
     
        return tf.image.adjust_saturation(image, saturation_factor), target, bbox

        
    