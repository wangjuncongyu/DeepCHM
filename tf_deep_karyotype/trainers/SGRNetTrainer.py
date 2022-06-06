# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:26:11 2022

@author: wjcongyu
"""
import os
import time
import numpy as np
import tensorflow as tf
from .base_trainer import BaseTrainer
from utils import rbbox as box_manager
from utils import helpers
from tensorflow.keras import backend as KB
from tensorflow.python.ops import math_ops

class SGRNetTrainer(BaseTrainer):
    def __init__(self, model, config, anchor_sampling_method=None, use_PLF=True, use_SGRA=True):
        super(SGRNetTrainer, self).__init__(model, config)
        self.anchor_sampling = anchor_sampling_method
        self.use_PLF = use_PLF
        self.use_SGRA = use_SGRA
        
    def start_train(self, train_dataset, val_dataset, save_model_dir, pretrained_file=None):
        '''
        Input:
            train_dataset: the data loader for training
            val_dataset: the dataset for online evaluation
            save_model_dir: the weights save path is osp.join(self.cfg.CHECKPOINTS_ROOT, save_model_dir)
            pretrain: if specified, then loading the pretrained weights
        '''
        self._print_config()
        self._prepare_path(save_model_dir)                
       
        self.model.build(input_shape = tuple([self.cfg.BATCH_SIZE]+self.cfg.INPUT_SHAPE + [1]))
        self.model.print_summary()
        
        # loading pretrained weights must behind model.build
        if not (pretrained_file is None): self._load_pretrained(pretrained_file)    
        
        print('\n----------------------------  start model training -----------------------------') 
        
        lr_schedule =  tf.keras.optimizers.schedules.ExponentialDecay(self.cfg.LR, self.cfg.DECAY_STEPS,\
                                                                      self.cfg.DECAY_RATE, staircase = True)
        
        optimizer =tf.keras.optimizers.Adam(learning_rate = lr_schedule)
       
        self.summary_writer = tf.summary.create_file_writer(self.save_path)
        with self.summary_writer.as_default(): 
        
            for self.epoch in range(self.cfg.EPOCHS):
                print ('\n################################### epoch:'+str(self.epoch+1)+'/'+str(self.cfg.EPOCHS))
               
                #training an epoch
                t1 = time.time()
                agv_salient_loss, agv_cls_loss, agv_reg_loss = self.__train_epoch(train_dataset, optimizer)
                t2 = time.time()
                current_lr = KB.eval(optimizer._decayed_lr('float32'))
                
               
                print ('\Salient loss: %f; Anchor cls loss: %f; Anchor reg loss: %f; Lr: %f; Used time (s): %f' % \
                      (agv_salient_loss, agv_cls_loss, agv_reg_loss, current_lr, t2-t1)) 
              
                tf.summary.scalar('salient map loss', agv_salient_loss, step = (self.epoch+1))
                tf.summary.scalar('anchor cls loss', agv_cls_loss, step = (self.epoch+1))
                tf.summary.scalar('anchor reg loss', agv_reg_loss, step = (self.epoch+1))
                #helpers.save_boxs2file('tmp/epoch_{0}.txt'.format(self.epoch), self._cur_pred_bboxs)
                self.checkpoint_file = os.path.join(self.save_path, "weights_epoch{0}.h5".format(self.epoch + 1))            
                print ('Saving weights to %s' % (self.checkpoint_file))
                self.model.save_weights(self.checkpoint_file)
                
                self._delete_old_weights(self.cfg.MAX_KEEPS_CHECKPOINTS) 
                
        print('\n---------------------------- model training completed ---------------------------')
               
        
    def __train_epoch(self, train_dataset, optimizer):
        '''
        train cfg.STEPS_PER_EPOCH (e.g., 100) in an epoch
        Input:
            train_dataset: the data loader for training
            optimizer: the optimizer for update the model parameters
        Output:
            salient_loss: the average loss of the salient map
            cls_loss: the average loss of the anchor classification net
            reg_loss: the average loss of the anchor regression net
            
        '''
        losses = {'salient_loss':[], 'cls_loss':[], 'reg_loss':[]}
       
        for step in range(self.cfg.STEPS_PER_EPOCH):
            bat_images, bat_salient_targets, bat_gt_rbboxes = train_dataset.next_batch(self.cfg.BATCH_SIZE) 
           
            with tf.GradientTape(persistent=False) as tape:
                
                bat_salient_preds, bat_ach_cls_scores, bat_ach_reg_deltas = self.model(self.__cvt2_feed_format(bat_images))    
                
                bat_ach_locations,\
                bat_anchors, \
                bat_ach_cls_labels, \
                bat_ach_weights, \
                bat_ach_reg_targets  = self.__generate_batch_anchor_targets(bat_salient_targets, \
                                                                                bat_gt_rbboxes, \
                                                                                bat_ach_cls_scores.numpy())
                
                #print(bat_salient_preds.shape, bat_salient_targets.shape)
                salient_loss = self.__calculate_salient_loss(bat_salient_targets, bat_salient_preds)
                losses['salient_loss'].append(salient_loss)
                
                ach_cls_loss = self.__calculate_ach_cls_loss(bat_ach_cls_labels, 
                                                             bat_ach_cls_scores,
                                                             bat_ach_locations, 
                                                             bat_ach_weights)
               
                losses['cls_loss'].append(ach_cls_loss)
               
                ach_reg_loss = self.__calculate_ach_reg_loss(bat_ach_reg_targets,
                                                             bat_ach_reg_deltas,
                                                             bat_ach_locations, 
                                                             bat_anchors,
                                                             bat_ach_weights,
                                                             bat_ach_cls_labels)
                losses['reg_loss'].append(ach_reg_loss)
                
                grad = tape.gradient(salient_loss + 2.0*ach_cls_loss + 4.0*ach_reg_loss, self.model.trainable_variables)
                   
                optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
                self._draw_progress_bar(step+1, self.cfg.STEPS_PER_EPOCH)
                
       
        return  tf.reduce_mean(losses['salient_loss']), tf.reduce_mean(losses['cls_loss']), tf.reduce_mean(losses['reg_loss'])
    
    def __calculate_salient_loss(self, salient_targets, salient_preds):
        '''
        the focal loss for keypoints (skeletons, connections, and intersecations) prediction
        Input:
            salient_targets: the ground-truth labels (i.e., gaussian targets, BxHxWx3)
            salient_preds: the predicted featuremap for the keypoints (BxHxWx3)
        Output:
            the scalar loss of the salient map prediction
        '''
        #print(location_gaus.shape, loc_preds.shape)
        salient_targets = math_ops.cast(salient_targets, salient_preds.dtype)
        salient_targets = tf.reshape(salient_targets, [-1])
        salient_preds = tf.reshape(salient_preds, [-1])
        
        num_pos = tf.reduce_sum(tf.cast(salient_targets == 1, tf.float32))
        #print ('num_pos:', num_pos)
        neg_weights = tf.math.pow(1 - salient_targets, 4)
        pos_weights = tf.ones_like(salient_preds, dtype=tf.float32)
        weights = tf.where(salient_targets == 1, pos_weights, neg_weights)
        inverse_preds = tf.where(salient_targets == 1, salient_preds, 1 - salient_preds)

        loss = tf.math.log(inverse_preds + 0.0001) * tf.math.pow(1 - inverse_preds, 2) * weights
        
        return -tf.reduce_sum(loss) / (num_pos + 1)
       

    def __calculate_ach_cls_loss(self, ach_cls_labels, ach_cls_preds, ach_locations, ach_weights, alpha=2.0, gamma=4.0):
        '''
        compute the penalized focal loss for the anchor classification subnetwork
        Input:            
            ach_cls_labels: the category labels for each anchor (0: negative; 1: positive; -1:ignore)
            ach_cls_preds: the anchor classification map, [B, H, W, A, 2]
            ach_locations: numpy of the locations (x,y) with size [BxN, 2] with each row of (x, y)
            ach_weights: the penalized weights for the loss calculation, [BxNxA]
        Output:
            the scalar loss of the anchor classification task
        '''       
        #the loss is calculated based on the centers
        ach_locations = ach_locations[:, [0, 2, 1]]
        pts = ach_locations.tolist()
      
        ach_cls_preds = tf.gather_nd(ach_cls_preds, pts)
        ach_cls_preds = tf.reshape(ach_cls_preds, [-1, self.model.n_classes])
        #ach_cls_preds = KL.Softmax()(ach_cls_preds)
        ach_cls_labels = math_ops.cast(ach_cls_labels, ach_cls_preds.dtype)
     
        keep = tf.where(ach_cls_labels!=-1)
        ach_cls_preds = tf.squeeze(tf.gather(ach_cls_preds, keep))
        ach_cls_gts = tf.squeeze(tf.gather(ach_cls_labels, keep))
        ach_weights = tf.squeeze(tf.gather(ach_weights, keep))
       
        fl = self.__softmax_focal_loss(ach_cls_gts, ach_cls_preds, alpha, gamma)
        penalized_fl = tf.where(ach_cls_gts==1, fl * ach_weights, fl)
       
        return tf.reduce_mean(penalized_fl)*100.0
    
    def __calculate_ach_reg_loss(self, ach_reg_targets, ach_reg_preds, ach_locations, anchors, ach_weights, ach_cls_labels):
        '''
        compute the penalized KLD loss for the anchor regression subnetwork
        Input:
            ach_reg_targets: the rbbox regression targets, [BxNxA, 5] with each row of (Gx, Gy, Gw, Gh, Gθ)            
            ach_reg_preds: the anchor regression map, [B, H, W, A, 5]
            ach_locations: numpy of the locations (x,y) with size [BxN, 2] with each row of (x, y)
            anchors: anchors corresponding to the ach_locations, [BxNxA, 5] with each row of (Ax,Ay,Aw,Ah,Aθ)
            ach_weights: the penalized weights for the loss calculation, [BxNxA]
            ach_cls_labels: the category labels for each anchor (0: negative; 1: positive; -1:ignore)
        Output:
            the scalar loss of the anchor regression task
            
        '''                              
        ach_locations = ach_locations[:, [0, 2, 1]]
        pts = ach_locations.tolist()
      
        ach_reg_preds = tf.gather_nd(ach_reg_preds, pts)
        ach_reg_preds = tf.reshape(ach_reg_preds, [-1, 5])
        
        keep = tf.where(ach_cls_labels==1)
        ach_reg_preds = tf.squeeze(tf.gather(ach_reg_preds, keep))
        anchors = tf.cast(tf.squeeze(tf.gather(anchors, keep)), ach_reg_preds.dtype)
        ach_reg_targets = tf.cast(tf.squeeze(tf.gather(ach_reg_targets, keep)), ach_reg_preds.dtype)
        ach_weights = tf.squeeze(tf.gather(ach_weights, keep))
      
        pred_rboxs = self.__rbbox_transform_inv(anchors, ach_reg_preds)
        KL_loss = self.__KL_loss(pred_rboxs, ach_reg_targets)
        KL_loss = tf.reduce_mean(tf.reshape(KL_loss, (keep.shape[0], keep.shape[0])), axis=-1)*ach_weights
      
        return tf.cast(tf.reduce_mean(KL_loss)*10.0, tf.float32)
                
   
    def __generate_batch_anchor_targets(self,bat_salient_targets, bat_gt_rbboxs, bat_ach_cls_score):
        '''
        Generate labes for anchor classification, and generate bbox regression targets
        Inputs:            
            bat_salient_targets: the batch of salient targets used to generate anchor locations, [B, H, W, 3]
            bat_gt_rbboxs: the batch of ground truth  rbboxes, list: {[G1, 8],..., [GB, 8]}
            bat_ach_cls_score: the batch of predicted anchor cls scores, used for sampling negative anchors, [B, H, W, A, 2]
        Output:
            bat_ach_locations: numpy of the locations (x,y) with size [BxN, 3] with each row of (batchid, x, y )
            bat_anchors: numpy of anchors corresponding to the ach_locations, [BxNxA, 5] with each row of (Ax,Ay,Aw,Ah,Aθ)
            bat_ach_cls_labels: the labels for anchor classification, [BxNxA], 0:negative, 1:positive, -1:ignored
            bat_ach_weights: the penalized weights for loss calculation, [BxNxA]
            bat_ach_reg_targets: the rbbox regression targets, [BxNxA, 5] with each row of (Gx, Gy, Gw, Gh, Gθ)
        '''
        bat_ach_locations = []; bat_anchors = []; bat_ach_cls_labels = []; bat_ach_weights = []; bat_ach_reg_targets = []
        #generate target for each batch
        for i in range(bat_salient_targets.shape[0]):
            ach_locations, \
            anchors, \
            ach_cls_labels,\
            ach_weights, \
            ach_reg_targets = self.__generate_anchor_targets(bat_salient_targets[i], bat_gt_rbboxs[i], bat_ach_cls_score[i])
                                                            
            batch_id = np.array([i for k in range(ach_locations.shape[0])])[:, np.newaxis]
            ach_locations = np.hstack([batch_id, ach_locations])
            bat_ach_locations.append(ach_locations)
            
            bat_anchors.append(anchors)
            bat_ach_cls_labels.append(ach_cls_labels)
            bat_ach_weights.append(ach_weights)
            bat_ach_reg_targets.append(ach_reg_targets)
            
        return np.vstack(bat_ach_locations), np.vstack(bat_anchors), np.hstack(bat_ach_cls_labels), \
                np.hstack(bat_ach_weights), np.vstack(bat_ach_reg_targets) 
            
    def __generate_anchor_targets(self, salient_target, gt_rbboxs, ach_cls_score):
        '''
        Generate labes for anchor classification, and generate bbox regression targets
        Inputs:            
            salient_targets: salient targets used to generate anchor locations, [H, W, 3]
            gt_rbboxs: the ground truth  rbboxes, [G, 8]
            ach_cls_score: the predicted anchor cls scores, used for sampling negative anchors, [H, W, A, 2]
        Output:
            ach_locations: numpy of the locations (x,y) with size [N, 2] with each row of (x, y)
            anchors: numpy of anchors corresponding to the ach_locations, [NxA, 5] with each row of (Ax,Ay,Aw,Ah,Aθ)
            ach_cls_labels: the labels for anchor classification, [NxA], 0:negative, 1:positive, -1:ignored
            ach_weights: the penalized weights for loss calculation, [NxA]
            ach_reg_targets: the rbbox regression targets, [NxA, 5] with each row of (Gx, Gy, Gw, Gh, Gθ)
        '''
        loc_thres = self.cfg.LOCATION_THRES if self.use_SGRA else 0.0
      
        ach_locations = np.where(np.max(salient_target[:,:,0:2], axis=-1)>=loc_thres)
        ach_locations = np.hstack([ach_locations[1][:, np.newaxis], ach_locations[0][:, np.newaxis]])
        anchors = box_manager.generate_ranchors(ach_locations.copy(),\
                                                self.cfg.ANCHOR_BASE_SIZE, \
                                                self.cfg.ANCHOR_RATIOS,\
                                                self.cfg.ANCHOR_SCALES,\
                                                self.cfg.ANCHOR_DETA_THETA,\
                                                self.cfg.FEATURE_STRIDE)
      
      
        ys = ach_locations[:, 1]; xs = ach_locations[:, 0]
        
        ach_cls_score = ach_cls_score[ys, xs, ...].reshape((-1, 2))
      
        ious = box_manager.calculate_rbbox_iou(anchors, gt_rbboxs[:,0:5])  
        
        #compute penalized weights for each anchor based on their cross-overlap,
        #length, bend, and densely distributed information
        argmax_ious = ious.argmax(axis=-1)       
        max_ious = ious[np.arange(anchors.shape[0]), argmax_ious]
        
        ach_reg_targets = gt_rbboxs[argmax_ious]
        if self.use_PLF:#using penalized loss functions (PLF)
            iou_score = max_ious/(np.sum(ious, axis=-1)+1e-8)
            bend_score = ach_reg_targets[:, 5]
            inter_score = ach_reg_targets[:, 6]
            len_score = np.where(ach_reg_targets[:, 2]>ach_reg_targets[:, 3], \
                                 ach_reg_targets[:, 2]/ach_reg_targets[:, 3], \
                                 ach_reg_targets[:, 3]/ach_reg_targets[:, 2])
        
            sum_score = np.abs(iou_score*bend_score*len_score+3.0*inter_score)
            ach_weights = 1.0+np.float32(1-1.0/np.exp(sum_score))
        else:# if not using PLF, just set all weights to 1.0
            #print('not use PLF!!!!!!!!!!!!!!!!!!!!')
            ach_weights = np.ones((ach_reg_targets.shape[0]), dtype=np.float32)
        
        #assign labels to each anchor:0 for negative; 1 for positive; -1 for ignored
        ach_reg_targets = ach_reg_targets[:,0:5]  
        
        ach_cls_labels = np.full((anchors.shape[0]), -1)
        
        pos_inds = np.where(max_ious>self.cfg.ANCHOR_POS_IOU_THRES)[0]
        keep_pos_inds = self.__sampling_positives(pos_inds, ach_cls_score)
        keep_pos_inds = np.concatenate([keep_pos_inds, ious.argmax(axis=0)])
       
        neg_inds = np.where(max_ious<self.cfg.ANCHOR_NEG_IOU_THRES)[0]
        N_positives = len(keep_pos_inds)
        if self.anchor_sampling=='uniform':
            #print('uniform sampling @@@@@@@@@@@@@@@@@@@@')
            keep_neg_inds = np.random.choice(neg_inds, N_positives, False)
        elif self.anchor_sampling=='daas':   
            #print('daas sampling @@@@@@@@@@@@@@@@@@@@')
            keep_neg_inds = self.__sampling_negatives_daas(neg_inds, ach_cls_score, N_positives)
        elif self.anchor_sampling=='ohem':
            #print('ohem sampling @@@@@@@@@@@@@@@@@@@@')
            keep_neg_inds = self.__sampling_negatives_ohem(neg_inds, ach_cls_score, N_positives)
        elif self.anchor_sampling=='hnas':
            #print('hnas sampling @@@@@@@@@@@@@@@@@@@@')
            keep_neg_inds = self.__sampling_negatives_hnas(max_ious, N_positives)
  
        ach_cls_labels[keep_neg_inds] = 0       
        ach_cls_labels[keep_pos_inds] = 1      
    
        #print('pos/neg:', np.sum(ach_cls_labels==1), np.sum(ach_cls_labels==0))
        return ach_locations, anchors, ach_cls_labels, ach_weights, ach_reg_targets  
    
    
    def __sampling_positives(self, pos_inds, ach_cls_score):
        '''
        Sampling positive anchors if too many anchors, to avoid out of GPU memory.
        '''
        if len(pos_inds) <= self.cfg.ANCHOR_MAX_NUM_POSITIVES:
            return pos_inds
      
        keep_pos_inds = np.random.choice(pos_inds, self.cfg.ANCHOR_MAX_NUM_POSITIVES, False)
        return keep_pos_inds
    
    def __sampling_negatives_daas(self, neg_inds, ach_cls_score, N_positives):
        '''
        Sampling negative anchors based on anchor classification loss (i.e., The DAAS)
        '''
        N_negatives= 2*N_positives
        if len(neg_inds) <= N_negatives:#self.cfg.ANCHOR_NUM_NEGATIVES:
            return neg_inds
        
       
        labels = np.zeros_like(neg_inds)
        
        loss_as_weights = self.__softmax_focal_loss(labels, \
                                                    ach_cls_score[neg_inds])
        loss_as_weights = loss_as_weights.numpy()
        
        loss_as_weights = np.float32(1-1.0/np.exp(loss_as_weights*4))
        
        keep_neg_inds = np.random.choice(neg_inds, N_negatives, False,
                                         helpers.softmax(loss_as_weights))
        
        return keep_neg_inds
    
    def __sampling_negatives_ohem(self, neg_inds, ach_cls_score, N_positives):
        '''
        Sampling negative anchors based on anchor classification loss (i.e., The DAAS)
        '''
        N_negatives= 2*N_positives
        if len(neg_inds) <= N_negatives:#self.cfg.ANCHOR_NUM_NEGATIVES:
            return neg_inds
        
        
        labels = np.zeros_like(neg_inds)
        
        loss_as_weights = self.__softmax_focal_loss(labels, \
                                                    ach_cls_score[neg_inds])
        loss_as_weights = loss_as_weights.numpy() 
        
        arg_idx = np.argsort(loss_as_weights)[::-1]
        #print(loss_as_weights[arg_idx])
        keep_neg_inds = neg_inds[arg_idx[0:N_negatives]]
        return keep_neg_inds
    
    def __sampling_negatives_hnas(self, max_ious, N_positives):
        '''
        Sampling negative anchors based on anchor classification loss (i.e., The DAAS)
        '''
        N_negatives= 2*N_positives
        N_hard_neg = int(N_negatives*0.5)
        N_easy_neg = N_negatives-N_hard_neg
        
        hard_inds = np.where((max_ious>0.1)&(max_ious<self.cfg.ANCHOR_NEG_IOU_THRES))[0]
        easy_inds = np.where(max_ious<=0.1)[0]
        
        keep_hard_inds = np.random.choice(hard_inds, N_hard_neg, False)
        keep_easy_inds = np.random.choice(easy_inds, N_easy_neg, False)
        keep_neg_inds = np.concatenate([keep_hard_inds, keep_easy_inds])
        return keep_neg_inds
    
   
    def __softmax_focal_loss(self, labels, preds, alpha=2.0, gamma=4.0):
        '''
        compute focal loss based on softmax predictions
        return: the loss for each anchor, [l1, l2, l3,....] 
        '''
        num_cls = preds.shape[-1]

        model_out = tf.clip_by_value(preds, 1e-9, 1.0)
        onehot_labels = tf.squeeze(tf.cast(tf.one_hot(tf.cast(labels,tf.int32),num_cls),model_out.dtype)) 
        ce = tf.multiply(onehot_labels, -tf.math.log(model_out))
        weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        fl = tf.reduce_max(fl, axis=1)
      
        return fl  
    
    def __KL_loss(self, pred_rboxs, ach_reg_targets, tau=1.0):
        '''
        computing the KLD loss between predicted bboxs and their regression target bboxes
        copy from: https://github.com/yangxue0827/RotationDetectionv
        '''
        pred_rboxs = tf.cast(pred_rboxs, tf.float32)
        ach_reg_targets = tf.cast(ach_reg_targets, tf.float32)       
        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.__get_gaussian_param(pred_rboxs, ach_reg_targets)
        KL_distance = tf.reshape(self.__KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)
       
        KL_distance = tf.maximum(tf.math.log(KL_distance + 1.), 0.)
        KL_similarity = 1.0 / (KL_distance + tau)        
      
        KL_loss = (1.0 - KL_similarity)
        return KL_loss
    
    def __get_gaussian_param(self, boxes_pred, target_boxes, shrink_ratio=1.):
        '''
        converting predicted bboxes and their regression target bboxes to 2D gaussians
        copy from: https://github.com/yangxue0827/RotationDetection
        '''
        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1]) * shrink_ratio
        w1 = tf.reshape(w1, [-1, 1]) * shrink_ratio
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1]) * shrink_ratio
        w2 = tf.reshape(w2, [-1, 1]) * shrink_ratio
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180.0
        theta2 *= np.pi / 180.0

        sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2.0
        sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2.0
        sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2]) + tf.cast(tf.linalg.eye(
            2) * 1e-5, tf.float32)

        sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
        sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
        sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2]) + tf.cast(tf.linalg.eye(
            2) * 1e-5,tf.float32)

        mu1 = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 1, 2])
        mu2 = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 1, 2])

        mu1_T = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 2, 1])
        mu2_T = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 2, 1])
        return sigma1, sigma2, mu1, mu2, mu1_T, mu2_T
    
    def __KL_divergence(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
        '''
        compute the KLD between two 2D gaussians
        copy from: https://github.com/yangxue0827/RotationDetection
        '''
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        item1 = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(sigma2_square), sigma1_square))
        item2 = tf.linalg.matmul(tf.linalg.matmul(mu2-mu1, tf.linalg.inv(sigma2_square)), mu2_T-mu1_T)
        item3 = tf.math.log(tf.linalg.det(sigma2_square) / (tf.linalg.det(sigma1_square) + 1e-4))
        return (item1 + item2 + item3 - 2) / 2. 
    
    def __cvt2_feed_format(self, images):
        if len(images.shape) == 3:
            images = images.reshape(list(images.shape) + [1])
        return images
    
    
    def __rbbox_transform_inv(self, anchors, deltas):
        dx, dy, dw, dh, dt = tf.unstack(deltas, axis=1)
        ax, ay, aw, ah, at = tf.unstack(anchors, axis=1)
        px = dx * aw + ax; py = dy * ah + ay;
        pw = tf.exp(dw) * aw; ph = tf.exp(dh) * ah
        pt = dt * 180.0 / np.pi + at

        return tf.transpose(tf.stack([px, py, pw, ph, pt]))