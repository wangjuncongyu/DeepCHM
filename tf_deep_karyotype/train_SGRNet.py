# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:32:57 2022

@author: wjcongyu
"""

import argparse
from datasets.my_dataset import MyDataset
from datasets.my_data_loader import MyDataLoader
from cfgs.cfgs import cfg
from datasets.transforms import TransformCompose
from datasets.transforms import RandomFlip, RandomRotate
#from datasets.transforms import RandomBrightness, RandomContrast,RandomGamma,RandomSaturation
from models.SGRNet import SGRNet
from models.backbones import BackboneFactory
from trainers.SGRNetTrainer import SGRNetTrainer

def get_args():
    parser = argparse.ArgumentParser(description='Train the SGRANet for rotated chromosome detection')
    parser.add_argument('--use_pti', '-use_pti', type=int, default=1, help='Use prior template information for anchor cls and reg')
    parser.add_argument('--ach_samp', '-ach_samp', type=str, default='daas', help='The achor sampling strategy, optional: uniform, daas, ohem, and hnas')
    parser.add_argument('--use_plf', '-use_plf', type=int, default=1, help='Use the penalized loss function for anchor cls and reg')
    parser.add_argument('--use_sgra', '-use_sgra', type=int, default=1, help='Use the SGRA strategy for rotated anchor setting')
    parser.add_argument('--use_augm', '-use_augm', type=int, default=1, help='Use data augmentation, i.e., randomflip, rotate, color')
    parser.add_argument('--load', '-load', type=str, default='last', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='SGRA', help='Load model from a .h5 file')
    parser.add_argument('--dtheta', '-dtheta', type=int, default=15, help='Theta interval for setting rotated anchorss')
    parser.add_argument('--epoch', '-epoch', type=int, default=120, help='Epoch for training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='Base learning rate for training')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    transformer = None if args.use_augm!=1 else TransformCompose([RandomFlip(), \
                                                                   RandomRotate()])

    dataset = MyDataset(cfg.TRAIN_DATA_ROOT, cfg.TRAIN_BBOX_FILE, transformer)
                
    dataloader = MyDataLoader(dataset)

    assert (args.dtheta>=5 and args.dtheta<=60 and (180%args.dtheta)==0), 'dtheta must in range [5, 60], and 180%dtheta=0 '
    cfg.ANCHOR_DETA_THETA = args.dtheta
    cfg.EPOCHS = args.epoch
    cfg.LR = args.lr
   
    n_anchors = len(cfg.ANCHOR_RATIOS)*len(cfg.ANCHOR_SCALES) * (180//cfg.ANCHOR_DETA_THETA)  
    cfg.NCLASSES = dataset.get_ncategores()

    net = SGRNet(backbone=BackboneFactory.get_backbone(cfg.BACKBONE), n_classes=cfg.NCLASSES, n_anchors=n_anchors, use_pti = args.use_pti==1)    
   
    trainer = SGRNetTrainer(net, cfg, args.ach_samp, args.use_plf==1, args.use_sgra==1)
    trainer.start_train(dataloader, None, args.save_dir, pretrained_file=args.load)