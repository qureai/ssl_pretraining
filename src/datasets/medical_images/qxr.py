import os
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec
import pandas as pd
import hashlib
import cv2
import torch.nn.functional as F

QXR_LABELS = {
    'opacity': 0,
}

def _get_sha1mod8(s):
    """Return the sha1 hash of the string s modulo 5."""
    sha1mod8 = int(hashlib.sha1(s.encode()).hexdigest()[-1], 16) % 9
    if sha1mod8 == 0:
        return "val"
    return "train"

class QXR(VisionDataset):
    """QXR dataset."""
    # Dataset information.
    MAE_OUTPUT_SIZE = 256
    NUM_CLASSES = 1  # Change which class can be chosen

    def __init__(self, config, download: bool = False, train: bool = True) -> None:
        
        self.config = config
        self.images_path = os.path.join(self.config.data_root, 'images') # train images folder
        self.csv_path = os.path.join(self.config.data_root, 'csvs', 'training_v4_full_16-6-22.csv') # train csv file
        super().__init__(self.images_path)
        columns = pd.read_csv(self.csv_path, nrows=0).columns
        self.df = pd.read_csv(self.csv_path, usecols=[columns[0]]+ list(QXR_LABELS.keys()), index_col=0)
        images_present = os.listdir(self.images_path)
        # remove .png from image names
        images_present = [image[:-4] for image in images_present]
        self.df = self.df[self.df.index.isin(images_present)]
        # split into train, test based on sha1 hash of image name
        self.df['split'] = self.df.index.map(_get_sha1mod8)
        self.df = self.df[self.df['split'] == 'train'] if train else self.df[self.df['split'] == 'val']
        self.df = self.df.drop(columns=['split'])
        
        self.INPUT_SIZE = (self.config.dataset.image_size, self.config.dataset.image_size)
        self.PATCH_SIZE = (self.config.dataset.patch_size, self.config.dataset.patch_size)
        self.IN_CHANNELS = self.config.dataset.in_channels
        
        self.TRANSFORMS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
        ])
        
        
    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Any:
        fname = self.df.index[index] + '.png'
        # image = Image.open(os.path.join(self.root, fname)).convert('RGB')
        image = cv2.imread(os.path.join(self.root, fname))
        if image is None:
            # TODO: handle this better
            image = np.random.randint(0, 255, (1024, 1024, 3)).astype(np.uint8)
        image = self.TRANSFORMS(image)
        image = (image - image.min())/(image.max() - image.min() + 1e-6) # normalize
        # check if image is a numpy array
        if not isinstance(image, np.ndarray):
            image = image.float()
        # get value opacity column from index
        label = torch.tensor(list(self.df.iloc[index].values)).long()
        label = torch.max(label, torch.tensor(0)) # remove -100 values
        index = torch.tensor(index)
        # if self.config.model.name == 'sam_adapt':
        #     image = self.preprocess(image)
        #     return index, image, label
        return index, image, label

    @staticmethod
    def num_classes():
        return QXR.NUM_CLASSES

    @staticmethod
    def spec(config):
        INPUT_SIZE = config.dataset.image_size
        PATCH_SIZE = config.dataset.patch_size
        IN_CHANNELS = config.dataset.in_channels
        return [
            Input2dSpec(input_size=INPUT_SIZE, patch_size=PATCH_SIZE, in_channels=IN_CHANNELS),
        ]
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).to(x.device)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).to(x.device)
        # Change x to 0-255
        # TODO: Do this better to cover all cases
        x = x * 255.0
        x = x.permute(1,2,0)
        # Normalize colors
        x = (x - pixel_mean) / pixel_std

        # Pad
        x = x.permute(2,0,1)
        h, w = x.shape[-2:]
        padh = self.config.dataset.image_size - h
        padw = self.config.dataset.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
