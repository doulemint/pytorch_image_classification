#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from fvcore.common.checkpoint import Checkpointer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
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
    '/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp02_SCAug/predictions.npz',
    '/data/nextcloud/dbc2017/files/jupyter/me/pytorch_image_classification/outputs/imagenet/efficientnet-b5/exp02_MU_Aug/predictions.npz'
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

        targets = targets.to(device)
        gt.extend(targets)
    clf = RandomForestClassifier(n_estimators=10)
    scores = cross_val_score(clf, X, y, cv=5)
    scores.mean()
    # clf = clf.fit(X, gt)

if __name__ == '__main__':
    randomForest()
