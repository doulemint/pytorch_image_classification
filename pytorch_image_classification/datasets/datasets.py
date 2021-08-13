from typing import Tuple, Union

import pathlib
from PIL import Image
import torch
import torchvision
import yacs.config
import pandas as pd
import json,cv2

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from pytorch_image_classification import create_transform

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from itertools import chain

def get_files(root,mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    else:
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root + x, os.listdir(root)))
        # print("image_folders",image_folders)
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x + "/*"), image_folders))))
        # print("all_images",all_images)
        if mode == "val":
            print("loading val dataset")
        elif mode == "train":
            print("loading train dataset")
        else:
            raise Exception("Only have mode train/val/test, please check !!!")
        label_dict={}
        for file in tqdm(all_images):
            all_data_path.append(file)
            name=file.split(os.sep)[-2] #['', 'data', 'nextcloud', 'dbc2017', 'files', 'images', 'train', 'Diego_Rivera', 'Diego_Rivera_21.jpg']
            # print(name)
            if name not in label_dict:
                label_dict[name]=len(label_dict)
            labels.append(label_dict[name])
            # labels.append(int(file.split(os.sep)[-2]))
        print(label_dict)
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files

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
        if i not in label_map.keys():
            label_map[i]=len(label_map)
    print(label_map)
    return label_map

def get_img(imgsrc):
    im_bgr = cv2.imread(imgsrc)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb
def get_img2(imgsrc):
    img = np.asarray(Image.open(imgsrc))
    return img
    
class MyDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
        self.output_label = output_label
        
        if output_label == True:
            self.labels = self.df['label'].values
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]
          
        # img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['filename']))
        img  = get_img2("{}".format(self.df.loc[index]['filename']))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.output_label == True:
            return img, target
        else:
            return img

class Data(Dataset):
    def __init__(self, df: pd.DataFrame,label_map,configs, transforms=None):
        self.files = [configs.dataset.dataset_dir +"/"+ file for file in df["image"].values]
        self.y = df["label"].values.tolist()
        self.label_map=label_map
        self.transforms = transforms
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i])
        label = self.label_map[self.y[i]]
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
        if is_train:
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
                df = pd.read_csv(config.dataset.cvsfile_train)
                label_map={}
                if config.dataset.jsonfile:
                    with open(config.dataset.jsonfile, "r") as f:
                        label_map = json.load(f)
                else:
                    label_map = getLabelmap(df['label'])
                    # label_map = {int(v): k for k, v in label_map.items()}
                if config.dataset.cvsfile_test:
                    train_df = pd.read_csv(config.dataset.cvsfile_train)
                    valid_df = pd.read_csv(config.dataset.cvsfile_test)
                else:
                    train_df, valid_df = train_test_split(df, stratify=df["label"].values)
                train_transform = create_transform(config, is_train=True)    
                # label_map = {int(k): v for k, v in label_map.items()}
                val_transform = create_transform(config, is_train=False)
                train_ds = Data(train_df,label_map,config,train_transform)
                valid_ds = Data(valid_df,label_map,config,val_transform)
                return train_ds, valid_ds
            else:
                raise ValueError() 
        else:
            if config.dataset.type == 'df':
              df = pd.read_csv(config.dataset.cvsfile_train)
              label_map = getLabelmap(df['artist'])
              df = pd.read_csv(config.dataset.cvsfile_test)
              val_transform = create_transform(config, is_train=False)
              valid_ds = Data(df,label_map,config,val_transform)
              
              return valid_ds
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            val_transform = create_transform(config, is_train=False)
            val_dataset = torchvision.datasets.ImageFolder(dataset_dir / 'val',
                                                            transform=val_transform)
            return val_dataset
    else:
        raise ValueError()



