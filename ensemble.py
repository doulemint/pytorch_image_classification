#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from stack_ensemble import ShallowNetwork
from scipy.stats import mode

from fvcore.common.checkpoint import Checkpointer
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,create_dataset,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def main():
    config = load_config()

    npz_files = ['/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/xp02_RC_Aug/predictions.npz',
    '/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp02_CMAug/predictions.npz',
    '/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp02_SCAug/predictions.npz',
    '/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp02_MU_Aug/predictions.npz'
               ]
# '/content/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp_Aug6/predictions.npz'
    whole_set = list(powerset(npz_files))

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    probs=[[0]*len(np.load(npz_files[0])['preds'][0])]*len(np.load(npz_files[0])['preds'])
    device = torch.device(config.device)

    for files in whole_set:
        if len(files)<2:
            continue
        for f in files:
            print(f)
            probs+= np.load(f)['preds'] 

    
        loss_meter = AverageMeter()
        correct_meter = AverageMeter()

        gt=[]
        for data, targets in tqdm.tqdm(test_loader):

            targets = targets.to(device)
            gt.extend(targets)
        probs=torch.tensor(probs)
        gt=torch.tensor(gt)
        loss = test_loss(probs, gt)
        _, preds = torch.max(probs, dim=1)
        # pred_prob_all=F.softmax(outputs, dim=1)
        correct_ = preds.eq(gt).sum().item()
        correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)
        print("new acc: ",accuracy,"loss: ",loss,"preds: ",preds)

def randomForest():
    X=[]
    npz_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_test.npz'
               ]
    for f in npz_files:
        print(f)
        X.append(np.load(f)['preds'])
    X=np.concatenate(X,axis=1)
    config = load_config()
    test_loader = create_dataloader(config, is_train=False)
    gt=[]
    device = torch.device(config.device)

    for _, targets in tqdm.tqdm(test_loader):

        # targets = targets.to(device)
        gt.extend(targets.numpy())
    clf = RandomForestClassifier(n_estimators=10)
    # print(clf.score(X, gt))
    scores = cross_val_score(clf, X, gt, cv=5)
    print(scores.mean())
    # clf = clf.fit(X, gt)

def logitregression():
    X=[]
    npz_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_train.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_train.npz'
               ]
    npz_test_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_test.npz'
               ]
    for f in npz_files:
        # print(f)
        X.append(np.load(f)['preds'])
    X=np.concatenate(X,axis=1)
    print(X.shape)

    X_test=[]
    for f in npz_test_files:
        # print(np.load(f)['preds'].shape)
        X_test.append(np.load(f)['preds'])
        
    X_test=np.concatenate(X_test,axis=1)
    print(X_test.shape)

    config = load_config()
    data_root = config.dataset.dataset_dir
    batch_size=config.train.batch_size
    num_workers = 2
    # labeled_dataset = MyDataset(train_clean, data_root, transforms=create_transform(config, is_train=False),data_type=config.dataset.subname)
    train_dataset, val_dataset = create_dataset(config, True)
    labeled_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # test_dataset = MyDataset(test_clean, data_root, transforms=create_transform(config, is_train=False),data_type=config.dataset.subname)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    gt=[]
    device = torch.device(config.device)

    gt_test=[]
    for _, targets in tqdm.tqdm(test_dataloader):
        # targets = targets.to(device)
        gt_test.extend(targets.numpy())
    gt_test = np.array(gt_test)
    print(gt_test.shape)

    for _, targets in tqdm.tqdm(labeled_dataloader):
      if targets=='unknown':
        targets=0
      # targets = targets.to(device)
      gt.extend(targets.numpy())
    # print(len(gt),len(gt[0]))
    gt = np.array(gt)
    print(gt.shape)
    
    model = ShallowNetwork()#LogisticRegression()
    model.fit(X, gt)
    yhat = model.predict(X_test)
    acc = accuracy_score(gt_test, yhat)
    print("acc: ",acc)
    # clf = clf.fit(X, gt)

def voting():

    npz_files = [
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp010_sc/predictions_test.npz',
    '/content/pytorch_image_classification/outputs/wiki22/efficientnet-b5/exp09_MU/predictions_test.npz'
               ]

    labels = []
    for f in npz_files:
        predicts = np.argmax(np.load(f)['preds'], axis=1)
        labels.append([predicts])
        
    # Ensemble with voting
    labels = np.array(labels)
    labels = np.transpose(labels, (1, 0))
    labels = mode(labels,axis=1)[0]
    
    labels = np.squeeze(labels)
    print(labels)

    config = load_config()
    _, test_loss = create_loss(config)
    test_loader = create_dataloader(config, is_train=False)
    gt=[]
    device = torch.device(config.device)
    correct_meter = AverageMeter()

    for _, targets in tqdm.tqdm(test_loader):

        targets = targets.to(device)
        gt.append([targets])

    # loss = test_loss(labels, gt)
    
    labels=torch.tensor(np.array(labels))
    print(labels,gt)
    correct_ = labels.eq(gt).sum().item()
    correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)
    print("new acc: ",accuracy,"preds: ",labels)

if __name__ == '__main__':
    # logitregression()
    randomForest()
    # voting()
    # main