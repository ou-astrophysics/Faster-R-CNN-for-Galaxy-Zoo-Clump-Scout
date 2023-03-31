#!/usr/bin/env python

# [START all]
import pandas as pd
import numpy as np
import os
import sys
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import cv2
# import seaborn as sns

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data.sampler import SequentialSampler
from torchvision.ops import nms

import transforms as T
import utils


# TORCH_DEVICE = 'mps' # there is currently a bug: https://github.com/pytorch/pytorch/issues/78915
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
    GPU_COUNT = torch.cuda.device_count()
    print('Device: {}, Number of GPUs: {}'.format(TORCH_DEVICE, GPU_COUNT))
else:
    TORCH_DEVICE = 'cpu'
    GPU_COUNT = 1

MODEL_PATH = './models_final/'
BATCH_SIZE = GPU_COUNT * 4


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


def get_model(model_type, model_load_path, device):
    """
    Creates the model object for Faster R-CNN
    and loads the corresponding checkpoint

    Args:
      model_load_path: full path to checkpoint
      device: torch device, 'cpu', 'cuda' or 'mps'

    Returns:
      FasterRCNN model

    """

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)

    # using DataParallel as model was saved as such
    model = nn.DataParallel(model)
    model = model.to(TORCH_DEVICE)

    model.load_state_dict(torch.load(
        model_load_path,
        map_location=torch.device(device)
    ))

    print('Model {} with checkpoint loaded'.format(model_type))
   
    return model


def get_transform():
    augs = []

    augs.append(T.PILToTensor())
    augs.append(T.ConvertImageDtype(torch.float))
    
    return T.Compose(augs)


def get_dataloader_dict(df, image_dir, is_colour):
    from GalaxyDataset import GalaxyDataset
    image_datasets = {}

    image_datasets = GalaxyDataset(
        dataframe=df,
        image_dir=image_dir,
        colour=is_colour,
        transforms=get_transform()
    )
    
    return torch.utils.data.DataLoader(
        image_datasets, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        collate_fn=utils.collate_fn
    )


def main():
    # [START Parse arguments from cmd]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_id', help='Select a model from resnet, resnet_trainable, zoobot_clumps, zoobot, zoobot_trainable.', type=str)
    parser.add_argument('--path', dest='image_path', help='Folder containing the galaxy cutouts.', type=str)
    parser.add_argument('--run', dest='run', nargs='?', default=20, help='Select checkpoint from specific run.', type=int)

    args = parser.parse_args()

    image_path = os.path.join(args.image_path, '')
    # [END Parse arguments from cmd]

    # [START create dataframe from image directory]
    df = (
        pd
        .DataFrame({
            'file_name': os.listdir(image_path), 
            'local_ids': [os.path.splitext(x)[0] for x in os.listdir(image_path)]
        })
        .assign(local_ids = lambda df_: df_.local_ids.astype(int))
        .assign(label = np.nan)
        .assign(label_text = '')
        .assign(x1 = np.nan)
        .assign(x2 = np.nan)
        .assign(y1 = np.nan)
        .assign(y2 = np.nan)
    )
    # [END create dataframe from image directory]

    # [START Inference]
    # initialise result-list
    results = []
    
    # load data
    dataloader = get_dataloader_dict(
        df=df,
        image_dir=image_path,
        is_colour=True
    )
    
    model_load_path = MODEL_PATH + model_dict[args.model_id]['model_name'] + '_run{:02d}'.format(args.run) + '_{}.pth'.format(model_dict[args.model_id]['epoch'])
    
    frcnn_model = get_model(
        model_type=model_dict[args.model_id]['model_type'],
        model_load_path=model_load_path,
        device=TORCH_DEVICE
    )
    
    # put the model in evaluation mode
    frcnn_model.eval()
    
    print('Running detections....')
    for images, targets in tqdm.tqdm(dataloader):
        images = list(img.to(TORCH_DEVICE) for img in images)
    
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        outputs = frcnn_model(images)
        outputs = [{k: v.to(TORCH_DEVICE) for k, v in t.items()} for t in outputs]
    
        with torch.no_grad():
            res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
    
            for k, v in res.items():
                results.append([
                    args.run,
                    model_dict[args.model_id]['model_name'],
                    k,
                    v['boxes'].data.cpu().numpy(),
                    v['labels'].data.cpu().numpy(),
                    v['scores'].data.cpu().numpy()
                ])
    
    df_results = (
        pd.DataFrame(
            results, 
            columns=['run', 'model_name', 'local_ids', 'boxes', 'labels', 'scores']
            )
        .explode(['boxes', 'labels', 'scores'])
        .dropna() # removing non-detections
        .assign(labels = lambda df_: df_.labels.astype(int))
        .assign(scores = lambda df_: df_.scores.astype(float))
    )
    # [END Inference]

    # [START Non-maximum suppression NMS]
    print('Running non-maximum suppression....')
    results = []
    _df = df_results.dropna()
    
    for local_id in tqdm.tqdm(_df['local_ids'].unique()):
        boxes = torch.from_numpy(np.vstack(_df[_df['local_ids'] == local_id]['boxes'])).to(TORCH_DEVICE)
        labels = torch.tensor(_df[_df['local_ids'] == local_id]['labels'].values, dtype=torch.int32).to(TORCH_DEVICE)
        scores = torch.tensor(_df[_df['local_ids'] == local_id]['scores'].values, dtype=torch.float32).to(TORCH_DEVICE)
    
        keep = nms(boxes = boxes, scores = scores, iou_threshold=0.2)
    
        results.append([
            args.run, 
            model_dict[args.model_id]['model_name'], 
            local_id, 
            boxes[keep].data.cpu().numpy(), 
            labels[keep].data.cpu().numpy(), 
            scores[keep].data.cpu().numpy()
        ])
    
    df_results = (
        pd.DataFrame(
            results, 
            columns=['run', 'model_name', 'local_ids', 'boxes', 'labels', 'scores']
            )
        .explode(['boxes', 'labels', 'scores'])
    )

    df_results[['px_x1', 'px_y1', 'px_x2', 'px_y2']] = pd.DataFrame(df_results['boxes'].tolist(), index= df_results.index)
    # [END Non-maximum suppression NMS]

    # [START Write output]
    result_file = 'FRCNN_preds_{}_run_{:02d}.gzip'.format(args.model_id, args.run)
    df_results.to_parquet(result_file, compression='gzip')
    print('Results written to {}'.format(result_file))
    # [END Write output]


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)

# [END all]