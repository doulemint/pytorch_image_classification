from typing import Tuple, Union

import pathlib

import torch
import torchvision
import yacs.config
import pandas as pd
import json

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from pytorch_image_classification import create_transform


class SubsetDataset(Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)

def getLabelmap(label_list):
    label_map={}
    for i in label_list:
        if i not in label_map.keys:
            label_map[i]=len(label_map)
    return label_map

class Data(Dataset):
    def __init__(self, df: pd.DataFrame,configs, transforms=None):
        self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["image_id"].values]
        self.y = df["label"].values.tolist()
        self.transforms = transforms
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i])
        label = self.y[i]
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, label

def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'CIFAR10',
            'CIFAR100',
            'MNIST',
            'FashionMNIST',
            'KMNIST',
    ]:
        module = getattr(torchvision.datasets, config.dataset.name)
        if is_train:
            if config.train.use_test_as_val:
                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = module(config.dataset.dataset_dir,
                                       train=is_train,
                                       transform=train_transform,
                                       download=True)
                test_dataset = module(config.dataset.dataset_dir,
                                      train=False,
                                      transform=val_transform,
                                      download=True)
                return train_dataset, test_dataset
            else:
                dataset = module(config.dataset.dataset_dir,
                                 train=is_train,
                                 transform=None,
                                 download=True)
                val_ratio = config.train.val_ratio
                assert val_ratio < 1
                val_num = int(len(dataset) * val_ratio)
                train_num = len(dataset) - val_num
                lengths = [train_num, val_num]
                train_subset, val_subset = torch.utils.data.dataset.random_split(
                    dataset, lengths)

                train_transform = create_transform(config, is_train=True)
                val_transform = create_transform(config, is_train=False)
                train_dataset = SubsetDataset(train_subset, train_transform)
                val_dataset = SubsetDataset(val_subset, val_transform)
                return train_dataset, val_dataset
        else:
            transform = create_transform(config, is_train=False)
            dataset = module(config.dataset.dataset_dir,
                             train=is_train,
                             transform=transform,
                             download=True)
            return dataset
    elif config.dataset.name == 'ImageNet':
        if config.dataset.type == 'dir':
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            train_transform = create_transform(config, is_train=True)
            val_transform = create_transform(config, is_train=False)
            train_dataset = torchvision.datasets.ImageFolder(
                dataset_dir / 'train', transform=train_transform)
            val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                        transform=val_transform)
            return train_dataset, val_dataset
        elif config.dataset.type == 'df':
            df = pd.read_csv(config.dataset.cvsfile)
            label_map={}
            if config.dataset.jsonfile:
                with open(config.dataset.jsonfile, "r") as f:
                    label_map = json.load(f)
            else:
                label_map = getLabelmap(df['label'])
                label_map = {int(v): k for k, v in label_map.items()}
            label_map = {int(k): v for k, v in label_map.items()}
            train_df, valid_df = train_test_split(df, stratify=df["label"].values)
            train_transform = create_transform(config, is_train=True)
            val_transform = create_transform(config, is_train=False)
            train_ds = Data(train_df, config,train_transform)
            valid_ds = Data(valid_df,config, val_transform)
            return train_ds, valid_ds
        else:
           raise ValueError() 
    else:
        raise ValueError()



