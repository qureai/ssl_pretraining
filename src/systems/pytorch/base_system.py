import os
from abc import abstractmethod

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.utils.data as data

from src.datasets.catalog import DATASET_DICT
from src.models.pytorch.transformer import DomainAgnosticTransformer
from src.models.pytorch.sam_adapter_encoder import ImageEncoderViT
from src.systems.pytorch.utils import DistributedProxySampler


def get_model(config: DictConfig, dataset_class: Dataset):
    '''Retrieves the specified model class, given the dataset class.'''
    if config.model.name == 'transformer':
        model_class = DomainAgnosticTransformer
        model = model_class(
            input_specs=dataset_class.spec(config),
            **config.model.kwargs,
        )
        return model
    elif config.model.name == 'sam_adapt':
        model_class = ImageEncoderViT
        model = model_class(**config.model.kwargs)
        sam_ckpt = "/home/users/sai.kiran/workspace/projects/MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth"
        ckpt = torch.load(sam_ckpt)
        for name, param in ckpt.items():
            if name.startswith('image_encoder'):
                model.load_state_dict({name.split('.',1)[1] : param}, strict=False)
        return model
    else:
        raise ValueError(f'Encoder {config.model.name} doesn\'t exist.')

    # Retrieve the dataset-specific params.
    return model_class(
        input_specs=dataset_class.spec(config),
        **config.model.kwargs,
    )


class BaseSystem(pl.LightningModule):

    def __init__(self, config: DictConfig):
        '''An abstract class that implements some shared functionality for training.

        Args:
            config: a hydra config
        '''
        super().__init__()
        self.config = config
        self.dataset = DATASET_DICT[config.dataset.name]
        self.dataset_name = config.dataset.name
        self.model = get_model(config, self.dataset)
        self.save_hyperparameters()

    @abstractmethod
    def objective(self, *args):
        '''Computes the loss and accuracy.'''
        pass

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def setup(self, stage):
        '''Called right after downloading data and before fitting model, initializes datasets with splits.'''
        self.train_dataset = self.dataset(config=self.config, download=True, train=True)
        self.val_dataset = self.dataset(config=self.config, download=True, train=False)
        try:
            print(f'{len(self.train_dataset)} train examples, {len(self.val_dataset)} val examples')
        except TypeError:
            print('Iterable/streaming dataset- undetermined length.')

    def train_dataloader(self):
        sampler = data.WeightedRandomSampler(
            weights=[1]*self.train_dataset.__len__(), num_samples=self.config.trainer.train_samples, replacement=True
        )
        # if self.config.trainer.strategy == "ddp":
        #     sampler = DistributedProxySampler(sampler)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            # shuffle=not isinstance(self.train_dataset, IterableDataset),
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise ValueError('Cannot get validation data for this dataset')
        # self.val_dataset.df has a column "opacity". Make weights array based on inverse number of examples.
        weights_dict = self.val_dataset.df['opacity'].value_counts().to_dict()
        weights = self.val_dataset.df['opacity'].apply(lambda x: 1/weights_dict[x]).tolist()

        sampler = data.WeightedRandomSampler(
            weights, num_samples=self.config.trainer.val_samples, replacement=False
        )
        # if self.config.trainer.strategy == "ddp":
        #     sampler = DistributedProxySampler(sampler)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            sampler=sampler,
            num_workers=self.config.dataset.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.config.optim.name == 'adam':
            optim = torch.optim.AdamW(params, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.name == 'sgd':
            optim = torch.optim.SGD(
                params,
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                momentum=self.config.optim.momentum,
            )
        else:
            raise ValueError(f'{self.config.optim.name} optimizer unrecognized.')
        if self.config.optim.lr_scheduler == 'onecyclelr':
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=self.config.optim.lr,
                anneal_strategy="cos",
                div_factor=25,
                steps_per_epoch=1,  # int((epoch_size / (bsize * acc_size))),
                epochs=self.config.trainer.max_epochs,
            )
        elif self.config.optim.lr_scheduler == 'constant':
            return optim
        return [optim], [sched]

    def on_train_end(self):
        model_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'model.ckpt')
        torch.save(self.state_dict(), model_path)
        print(f'Pretrained model saved to {model_path}')
