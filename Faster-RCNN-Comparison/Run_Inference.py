#!/usr/bin/env python

# [START all]

import logging
import pandas as pd
import numpy as np
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import sys
sys.path.append('/home/fortson/jpopp/shared/')
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler

import transforms as T
import utils


if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
    GPU_COUNT = torch.cuda.device_count()
    print('Device: {}, Number of GPUs: {}'.format(TORCH_DEVICE, GPU_COUNT))
else:
    TORCH_DEVICE = 'cpu'
    GPU_COUNT = 1

DATA_PATH = '/home/fortson/hdickins/clumpy_data/'
IMAGE_PATH = DATA_PATH + 'real_pngs/'
PRE_TRAINED_MODELS_PATH = '/home/fortson/jpopp/pre_trained_models/'
FINAL_MODEL_PATH = '/home/fortson/jpopp/Faster_R-CNN_Comparison/models_final/'

NUM_CLASSES = 3 # 2 classes (clump, odd clump) + background

BATCH_SIZE = GPU_COUNT * 8
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


def get_dataloader_dict(df, image_dir, cutout, is_colour):
    import SDSSGalaxyDataset
    image_datasets = {}

    image_datasets = SDSSGalaxyDataset.SDSSGalaxyDatasetInference(
        dataframe=df,
        image_dir=image_dir,
        cutout=cutout,
        colour=is_colour,
        transforms=get_transform(train=False)
    )
    
    return torch.utils.data.DataLoader(
        image_datasets, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=utils.collate_fn
    )


def main():
    # Parse arguments from cmd
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
            'epoch' : 20,
        },
        'resnet_trainable' : {
            'model_type' : 'Resnet_Imagenet',
            'model_name' : 'Resnet_Imagenet_trainable',
            'trainable_layers' : 3,
            'description' : 'ResNet50 initialised with default weights IMAGENET1K_V1 and the last 3 blocks made trainable',
            'epoch' : 20,
        },
        'zoobot_clumps' : {
            'model_type' : 'Zoobot_fine_tuned',
            'model_name' : 'Zoobot_fine_tuned',
            'trainable_layers' : 0,
            'description' : 'ResNet50 initialised with weights from a Zoobot classifier fine-tuned for clumps',
            'epoch' : 120,
        },
        'zoobot' : {
            'model_type' : 'Zoobot_pre_trained',
            'model_name' : 'Zoobot_pre_trained',
            'trainable_layers' : 0,
            'description' : 'ResNet50 initialised with weights from Zoobot, all layers kept fix for training',
            'epoch' : 120,
        },
        'zoobot_trainable' : {
            'model_type' : 'Zoobot_pre_trained',
            'model_name' : 'Zoobot_pre_trained_trainable',
            'trainable_layers' : 3,
            'description' : 'ResNet50 initialised with weights from Zoobot, last 3 blocks made trainable',
            'epoch' : 120,
        },
    }
    # [END model definition]

    # [START input data prepartion]
    # read the full set with bounding boxes
    df = pd.read_pickle('clump_scout_full_set.pkl').rename(columns={'local_id': 'local_ids'})
    
    df = df[['zoo_id', 'local_ids', 'label', 'label_text', 'x1', 'x2', 'y1', 'y2']]
    df['local_ids'] = df['local_ids'].astype(int)
    df['label'] = df['label'].astype(int)
    # [END input data prepartion]

    # [START inference]
    # load data
    dataloader = get_dataloader_dict(
        df=df,
        image_dir=IMAGE_PATH,
        cutout=CUTOUT,
        is_colour=True
    )

    # empty GPU cache
    torch.cuda.empty_cache() 

    # initialise result-list
    results = []
    
    for run in range(args.run_from, args.run_to, 1):
        print('Executing run: {}'.format(run))
        
        for model, model_data in model_dict.items():
            # get the model        
            model_load_path = FINAL_MODEL_PATH + model_data['model_name'] + '_run{:02d}'.format(run) + '_{}.pth'.format(model_data['epoch'])
            print('Model checkpoint loaded from here: {}'.format(model_load_path))
            
            frcnn_model = get_model(
                model_type=model_data['model_type'],
                num_classes=NUM_CLASSES,
                trainable_layers=model_data['trainable_layers']
            )
            
            # using DataParallel as model was saved as such
            frcnn_model = nn.DataParallel(frcnn_model)
            frcnn_model = frcnn_model.to(TORCH_DEVICE)
            
            # Zoobot_pre_trained_run20_120.pth
            frcnn_model.load_state_dict(torch.load(
                model_load_path,
                map_location=torch.device(TORCH_DEVICE)
            ))
            
            # put the model in evaluation mode
            frcnn_model.eval()
            
            for images, targets in dataloader:
                images = list(img.to(TORCH_DEVICE) for img in images)
            
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                outputs = frcnn_model(images)
                outputs = [{k: v.to(TORCH_DEVICE) for k, v in t.items()} for t in outputs]
            
                with torch.no_grad():
                    res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
            
                    for k, v in res.items():
                        results.append([
                            run,
                            model_data['model_name'],
                            k,
                            v['boxes'].data.cpu().numpy(),
                            v['labels'].data.cpu().numpy(),
                            v['scores'].data.cpu().numpy()
                        ])
        
            with torch.no_grad():
                torch.cuda.empty_cache()
    
    # write final results
    df_results = (
        pd.DataFrame(
            results, 
            columns=['run', 'model_name', 'local_ids', 'boxes', 'labels', 'scores']
            )
        .explode(['boxes', 'labels', 'scores'])
    )
    df_results.to_parquet('FRCNN_ClumpScout_Predictions.gzip', compression='gzip')  
    # [END inference]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)

# [END all]