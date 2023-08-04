from pytorch_lightning import Callback, LightningModule, Trainer
from src.datasets.catalog import DATASET_DICT
import torch
import numpy as np
from einops.layers.torch import Rearrange
import torch.nn as nn
import cv2
import os

class PretrainingOutputVisualiser(Callback):
    " Generates MAE output for a model and saves it as an image for a random set of 10 images "
    def __init__(self, config):
        super().__init__()
        self.dataset_class = DATASET_DICT[config.dataset.name]
        self.config = config
        self.dataset = self.dataset_class(config=self.config, download=True, train=False)
        np.random.seed(self.config.trainer.seed)
        self.ids = np.random.randint(0, self.dataset.__len__(), 10)
        # save the images in a folder in self.exp.base_dir in a folder named output_visualisation and inner folder name as original
        # save images as image_number.png in the original folder
        batch = [self.dataset[i] for i in self.ids]
        batch = torch.stack([b[1] for b in batch])
        self.batch = batch
        batch = batch.permute(0, 2, 3, 1) # (b, h, w, c)
        batch = batch.detach().cpu().numpy()
        self.save_images(batch, 'original') 
        
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # get a random set of 10 images from dataset
        # get the output of the model for each of these images
        batch = self.batch
        with torch.no_grad():
            batch = batch.to(pl_module.device)
            pl_module.eval()
            representations = pl_module.model.forward([batch], prepool=True)
            # remove 0th element in 2nd dimension
            if pl_module.hparams.config.model.name == 'sam_adapt':
                representations = representations
            else:
                representations = representations[:, 1:, :]
            preds = pl_module.patch_transform(representations)
            preds = pl_module.decoder(preds)
            pl_module.train()
        
        p = patch_size = self.config.dataset.patch_size
        image_size = self.config.dataset.image_size
        n = num_patches_pre_row = image_size//patch_size
        c = channels = self.config.dataset.in_channels
        reshape_to_image = nn.Sequential(
                Rearrange(
                    'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=n, w=n, c=c
                )
            )
        # im_preds = reshape_to_image(preds) # (b, c, h, w) - range 0-1
        im_preds = preds.permute(0, 2, 3, 1) # (b, h, w, c)
        im_preds = im_preds.detach().cpu().numpy()
        # dir_path = os.path.join(self.config.exp.base_dir, self.config.exp.name, 'output_visualisation', str(trainer.current_epoch))
        # os.makedirs(dir_path, exist_ok=True)
        # np.save(os.path.join(dir_path, 'preds.npy'), im_preds)
        im_preds = np.clip(im_preds, 0, 1)
        self.save_images(im_preds, str(trainer.current_epoch))
        
        
            
    def save_images(self, images, folder_name: str):
        # images must be already detached and in cpu and numpy in (b h w c) format and 0-1 range
        for i, im in enumerate(images):
            # im = (im - im.min())/(im.max() - im.min())
            im = (im*255).astype(np.uint8)
            dir_path = os.path.join(self.config.exp.base_dir, self.config.exp.name, 'output_visualisation', folder_name)
            os.makedirs(dir_path, exist_ok=True)
            # save image
            cv2.imwrite(os.path.join(dir_path, f'{i}.png'), im)
        