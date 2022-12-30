import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np
import pandas as pd
from PIL import Image


class SDSSGalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, cutout=(50, 50, 350, 350), transforms=None):
        # super().__init__()
        self.df = dataframe
        self.image_dir = image_dir
        self.cutout = cutout
        self.transforms = transforms
        self.image_ids = dataframe['local_ids'].unique()
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['local_ids'] == image_id]

        image = Image.open(f"{self.image_dir}/{image_id}.png").convert('RGB') # remove 4th channel of PNGs
        image = image.convert('L') # grayscale
        image = image.crop(self.cutout)
        image = np.asarray(image).astype(np.float32)
        image /= 255.0
        image = torch.as_tensor(image, dtype=torch.float32)
        image = F.to_pil_image(image, mode='L')
        
        # Handle empty bounding boxes
        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        if np.isnan(boxes).all():
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # Labels - need to get them into the right
        labels_list = records[['label']].values
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        labels = torch.squeeze(labels, 1)
        
        # no crowd instances
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]
