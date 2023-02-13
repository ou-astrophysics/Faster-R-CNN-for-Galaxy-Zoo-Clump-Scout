#!/usr/bin/env python

# [START all]

import logging
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./shared/')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import cv2
import pytorch_lightning as pl
import time

from torchsummary import summary
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.model_selection import train_test_split

import transforms as T
import utils
import engine


if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
    GPU_COUNT = torch.cuda.device_count()
    print('Device: {}, Number of GPUs: {}'.format(TORCH_DEVICE, GPU_COUNT))
else:
    TORCH_DEVICE = 'cpu'
    GPU_COUNT = 1

DATA_PATH = './'
IMAGE_PATH = DATA_PATH + 'real_pngs/'
PRE_TRAINED_MODELS_PATH = './pre_trained_models/'

NUM_CLASSES = 3 # 2 classes (clump, odd clump) + background
NUM_EPOCHS = 120

BATCH_SIZE = GPU_COUNT * 4
CUTOUT = (100, 100, 300, 300)
CUTOUT_ARRAY = np.array([100, 300, 100, 300])


def get_model(model_type, num_classes=3, trainable_layers=0):
    """
    Creates the model object for Faster R-CNN

    Args:
      model_name (str): 'Zoobot_pre_trained', 'Zoobot_fine_tuned', 'Resnet_Imagenet'
      num_classes (int): number of classes the detector should outpub, 
        must include a class for the background
      trainable_layers (int): number of blocks of the classification backbone,
        counted from top, that should be made trainable
        e.g. 0 - all blocks fixed, 1 - 'backbone.body.conv1' trainable

    Returns:
      FasterRCNN model

    """
    import copy_zoobot_weights

    # load an object detection model pre-trained on COCO, all layers fixed
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights_backbone='IMAGENET1K_V1',
        trainable_backbone_layers=trainable_layers
    )

    try:
        if model_type == 'Zoobot_pre_trained':
            # zoobot_ckpt_path = PRE_TRAINED_MODELS_PATH + 'Zoobot_Resnet_Torchvision/epoch=20-step=6552.ckpt'
            zoobot_ckpt_path = PRE_TRAINED_MODELS_PATH + 'Zoobot_Resnet_Torchvision_evo/epoch=88-step=51353.ckpt'
            model = copy_zoobot_weights.copy_Zoobot_weights_to_Resnet(
                model=model, 
                ckpt_path=zoobot_ckpt_path,
                device=TORCH_DEVICE,
                trainable_layers=trainable_layers
            )
            print('Zoobot pre-trained loaded.')
    
        elif model_type == 'Zoobot_fine_tuned':
            # zoobot_ckpt_path = PRE_TRAINED_MODELS_PATH + 'Zoobot_Clumps_Resnet/Zoobot_Clump_Classifier_36.pth'
            zoobot_ckpt_path = PRE_TRAINED_MODELS_PATH + 'Zoobot_Clumps_Resnet/Zoobot_Clump_Classifier_new_21.pth'
            model = copy_zoobot_weights.copy_Zoobot_clumps_weights_to_Resnet(
                model=model, 
                ckpt_path=zoobot_ckpt_path,
                device=TORCH_DEVICE,
                trainable_layers=trainable_layers
            )
            print('Zoobot fine-tuned for clumps loaded.')
        
        elif model_type == 'Resnet_Imagenet':
            print('ResNet initialised with Imagenet weights loaded.')
    
        else:
            print('None of the valid models chosen.')
    
    except Exception as e:
        print(str(e))

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
    return model


def get_transform(train):
    augs = []

    augs.append(T.PILToTensor())
    augs.append(T.ConvertImageDtype(torch.float))
    
    if train:
        augs.append(T.RandomHorizontalFlip(0.5))
        augs.append(T.RandomVerticalFlip(0.5))
    
    return T.Compose(augs)


def get_dataloader_dict(train_df, val_df, image_dir, cutout, is_colour):
    import SDSSGalaxyDataset
    image_datasets = {}

    image_datasets['train'] = SDSSGalaxyDataset.SDSSGalaxyDataset(
        dataframe=train_df,
        image_dir=image_dir,
        cutout=cutout,
        colour=is_colour,
        transforms=get_transform(train=True)
    )
    image_datasets['val'] = SDSSGalaxyDataset.SDSSGalaxyDataset(
        dataframe=val_df,
        image_dir=image_dir,
        cutout=cutout,
        colour=is_colour,
        transforms=get_transform(train=False)
    )
    
    return {x: torch.utils.data.DataLoader(
        image_datasets[x], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=utils.collate_fn
    ) for x in ['train', 'val']}


def main():
    # Parse arguments from cmd
    # python ~/Faster_R-CNN_Comparison/Run_model_comparison.py --from 11 --to 14
    parser = argparse.ArgumentParser()
    parser.add_argument('--from', dest='run_from', help='first run', type=int)
    parser.add_argument('--to', dest='run_to', help='last run', type=int)

    args = parser.parse_args()

    # [START model definition]
    # create model dict
    model_dict = {
        'resnet' : {
            'model_type' : 'Resnet_Imagenet',
            'model_name' : 'Resnet_Imagenet',
            'trainable_layers' : 0,
            'description' : 'ResNet50 initialised with default weights IMAGENET1K_V1',
            'log_dir' : './models/FRCNN_Resnet_Imagenet/',
        },
        'resnet_trainable' : {
            'model_type' : 'Resnet_Imagenet',
            'model_name' : 'Resnet_Imagenet_trainable',
            'trainable_layers' : 3,
            'description' : 'ResNet50 initialised with default weights IMAGENET1K_V1 and the last 3 blocks made trainable',
            'log_dir' : './models/FRCNN_Resnet_Imagenet_trainable/',
        },
        'zoobot_clumps' : {
            'model_type' : 'Zoobot_fine_tuned',
            'model_name' : 'Zoobot_fine_tuned',
            'trainable_layers' : 0,
            'description' : 'ResNet50 initialised with weights from a Zoobot classifier fine-tuned for clumps',
            'log_dir' : './models/FRCNN_Resnet_Zoobot_Clumps/',
        },
        'zoobot' : {
            'model_type' : 'Zoobot_pre_trained',
            'model_name' : 'Zoobot_pre_trained',
            'trainable_layers' : 0,
            'description' : 'ResNet50 initialised with weights from Zoobot, all layers kept fix for training',
            'log_dir' : './models/FRCNN_Resnet_Zoobot/',
        },
        'zoobot_trainable' : {
            'model_type' : 'Zoobot_pre_trained',
            'model_name' : 'Zoobot_pre_trained_trainable',
            'trainable_layers' : 3,
            'description' : 'ResNet50 initialised with weights from Zoobot, last 3 blocks made trainable',
            'log_dir' : './models/FRCNN_Resnet_Zoobot_trainable/',
        },
    }
    # [END model definition]

    # [START run prepartion]
    # read the image-to-run relation
    df_runs = pd.read_pickle('image_ids_for_runs_log.pkl')
    
    # read the full set with bounding boxes
    df = pd.read_pickle('clump_scout_full_set.pkl').rename(columns={'local_id': 'local_ids'})
    
    df = df[['zoo_id', 'local_ids', 'label', 'label_text', 'x1', 'x2', 'y1', 'y2']]
    df['local_ids'] = df['local_ids'].astype(int)
    df['label'] = df['label'].astype(int)

    runs = df_runs['run'].unique()
    groups = ['Training', 'Validation']
    
    df_data = (
        df_runs[df_runs['group'].isin(groups)]
        .merge(df, how='inner', on='zoo_id')
    )
    # [END run prepartion]

    # [START Training]
    for run in range(args.run_from, args.run_to, 1):
        print('Executing run: {}'.format(run))
        
        # load data
        dataloader_dict = get_dataloader_dict(
            train_df=df_data[(df_data['run']==run) & (df_data['group']=='Training')],
            val_df=df_data[(df_data['run']==run) & (df_data['group']=='Validation')],
            image_dir=IMAGE_PATH,
            cutout=CUTOUT,
            is_colour=True
        )
    
        for model, model_data in model_dict.items():
            # initialise Tensorboard writer
            tb_log_dir = model_data['log_dir'] + 'run={}/'.format(run) + 'train'
            writer = SummaryWriter(log_dir=tb_log_dir)
    
            # get the model
            frcnn_model = get_model(
                model_type=model_data['model_type'],
                num_classes=NUM_CLASSES,
                trainable_layers=model_data['trainable_layers']
            )
            
            # using all available GPUs and move model to the right device
            frcnn_model = nn.DataParallel(frcnn_model)
            frcnn_model = frcnn_model.to(TORCH_DEVICE)
            
            # construct an optimizer
            params = [p for p in frcnn_model.parameters() if p.requires_grad]
            # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.00005)
            
            # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
            # Looping through epochs
            for epoch in range(NUM_EPOCHS):
                # train for one epoch, printing every 10 iterations
                engine.train_one_epoch(
                    frcnn_model, 
                    optimizer, 
                    dataloader_dict['train'], 
                    TORCH_DEVICE, 
                    epoch, 
                    print_freq=10,
                    scaler=None,
                    tb_writer=writer
                    # tb_writer=None
                )
                
                # update the learning rate
                # lr_scheduler.step()
    
                # evaluation loss
                engine.evaluate_loss(
                    frcnn_model, 
                    dataloader_dict['val'], 
                    TORCH_DEVICE, 
                    epoch, 
                    tb_writer=writer
                    # tb_writer=None
                )
                
                # evaluate on the test dataset
                engine.evaluate(
                    frcnn_model, 
                    dataloader_dict['val'], 
                    device=TORCH_DEVICE,
                    epoch=epoch, 
                    tb_writer=writer
                    # tb_writer=None
                )
            
                if (epoch+1) % 20 == 0:
                    model_save_path = model_data['log_dir'] + 'run={}/'.format(run) + model_data['model_name'] + '_{}.pth'.format(epoch+1)
                    torch.save(frcnn_model.state_dict(), model_save_path)

            with torch.no_grad():
            	torch.cuda.empty_cache()
    # [END Training]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)

# [END all]